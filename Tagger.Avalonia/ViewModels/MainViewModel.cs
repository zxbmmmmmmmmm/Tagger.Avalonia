using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Avalonia.Platform.Storage;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CsvHelper;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using Tagger.Avalonia.Models;

namespace Tagger.Avalonia.ViewModels;

public partial class MainViewModel(IStorageProvider provider) : ViewModelBase
{
    [ObservableProperty]
    public partial string? ModelPath { get; set; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(Image))]
    public partial string? ImagePath { get; set; }

    [ObservableProperty]
    public partial string? TagsPath { get; set; }

    [ObservableProperty]
    public partial string? ConfigPath { get; set; }

    [ObservableProperty]
    public partial List<TagInfo>? GeneralTags { get; set; }

    [ObservableProperty]
    public partial List<TagInfo>? CharacterTags { get; set; }

    [ObservableProperty]
    public partial List<TagInfo>? RatingTags { get; set; }

    public IStorageProvider StorageProvider { get; init; } = provider;

    public Task<Bitmap?> Image => ImagePath is null?null: ImageHelper.LoadFromFileAsync(ImagePath);

    [RelayCommand]
    public async Task InferenceAsync()
    {
        if (ModelPath is null || ImagePath is null || TagsPath is null || ConfigPath is null) return;
        var result = await CaformerDbv4Inference.InferenceAsync(ModelPath, ImagePath, TagsPath, ConfigPath);
        (GeneralTags, CharacterTags, RatingTags) = result;
    }

    [RelayCommand]
    public async Task SelectModelPathAsync()
    {
        var file = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions { AllowMultiple = false, SuggestedFileName = "model.onnx" });
        if (file.Count is 0) return;
        ModelPath = file[0].TryGetLocalPath();
    }


    [RelayCommand]
    public async Task SelectImagePathAsync()
    {
        var file = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions { AllowMultiple = false });
        if (file.Count is 0) return;
        ImagePath = file[0].TryGetLocalPath();
    }

    [RelayCommand]
    public async Task SelectTagsPathAsync()
    {
        var file = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions { AllowMultiple = false, SuggestedFileName = "selected_tags.csv" });
        if (file.Count is 0) return;
        TagsPath = file[0].TryGetLocalPath();
    }

    [RelayCommand]
    public async Task SelectConfigPathAsync()
    {
        var file = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions { AllowMultiple = false, SuggestedFileName = "config.json" });
        if (file.Count is 0) return;
        ConfigPath = file[0].TryGetLocalPath();
    }
}
public static class ImageHelper
{
    public static async Task<Bitmap> LoadFromFileAsync(string path)
    {
        var stream = File.OpenRead(path);
        return await Task.Run(() => Bitmap.DecodeToWidth(stream, 1000));
    }

    public static async Task<Bitmap?> LoadFromWeb(Uri url)
    {
        using var httpClient = new HttpClient();
        try
        {
            var response = await httpClient.GetAsync(url);
            response.EnsureSuccessStatusCode();
            var data = await response.Content.ReadAsByteArrayAsync();
            return new Bitmap(new MemoryStream(data));
        }
        catch (HttpRequestException ex)
        {
            Console.WriteLine($"An error occurred while downloading image '{url}' : {ex.Message}");
            return null;
        }
    }
}

class CaformerDbv4Inference
{
    static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
    static readonly float[] Std = { 0.229f, 0.224f, 0.225f };
    const int PadSize = 512;
    const int FinalSize = 384;

    static DenseTensor<float> Preprocess(string imagePath)
    {
        using var image = Image.Load<Rgba32>(imagePath);

        int targetSize = Math.Max(Math.Max(image.Width, image.Height), PadSize);
        using var padded = new Image<Rgba32>(targetSize, targetSize, new Rgba32(255, 255, 255, 255));

        int offsetX = (targetSize - image.Width) / 2;
        int offsetY = (targetSize - image.Height) / 2;

        padded.Mutate(x => x.DrawImage(image, new Point(offsetX, offsetY), 1.0f));
        padded.Mutate(x => x.Resize(new ResizeOptions
        {
            Size = new Size(FinalSize, FinalSize),
            Mode = ResizeMode.Stretch,
            Sampler = KnownResamplers.Bicubic
        }));

        Rectangle cropRect = new(
            (padded.Width - FinalSize) / 2, (padded.Height - FinalSize) / 2,
            FinalSize, FinalSize
        );
        padded.Mutate(x => x.Crop(cropRect));

        var tensor = new DenseTensor<float>([1, 3, FinalSize, FinalSize]); // BCHW
        for (int y = 0; y < FinalSize; y++)
        {
            Span<Rgba32> rowSpan = padded.DangerousGetPixelRowMemory(y).Span;
            for (int x = 0; x < FinalSize; x++)
            {
                var px = rowSpan[x];
                float r = px.R / 255f; float g = px.G / 255f; float b = px.B / 255f;
                tensor[0, 0, y, x] = (r - Mean[0]) / Std[0]; tensor[0, 1, y, x] = (g - Mean[1]) / Std[1]; tensor[0, 2, y, x] = (b - Mean[2]) / Std[2];
            }
        }
        return tensor;
    }

    static List<TagRecord> LoadTagMeta(string csvPath)
    {
        using var reader = new StreamReader(csvPath);
        using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
        return csv.GetRecords<TagRecord>().ToList();
    }


    static void ShowInferenceResults(string imagePath, (List<(float, string)> general_tags, List<(float, string)> character_tags, List<(float, string)> rating_tags) d)
    {
        List<(float, string)> general_tags, character_tags, rating_tags;
        (general_tags, character_tags, rating_tags) = d;
        Console.WriteLine($"Tagged results for {Path.GetFileName(imagePath)}:");
        Console.WriteLine(" General Tags:");
        ShowResult(general_tags);
        Console.WriteLine(" Character Tags:");
        ShowResult(character_tags);
        Console.WriteLine(" Rating Tags:");
        ShowResult(rating_tags);
    }

    static void ShowResult(List<(float, string)> general_tags)
    {
        foreach (var (prob, label) in general_tags)
            Console.WriteLine($"  - {label} ({prob:P2})");
    }

    public static async Task<(List<TagInfo> general_tags, List<TagInfo> character_tags, List<TagInfo> rating_tags)>
        InferenceAsync(string modelPath, string imagePath, string tagsCsv, string configPath)
    {
        var inputTensor = Preprocess(imagePath);
        var tagMeta = LoadTagMeta(tagsCsv).ToDictionary(x => x.name);
        var timmConfig = JsonSerializer.Deserialize<TimmConfig>(await File.ReadAllTextAsync(configPath), new JsonSerializerOptions { TypeInfoResolver = SourceGenerationContext.Default })!;
        var outputMap = timmConfig.tags;
        using var session = new InferenceSession(modelPath);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };
        using var results =await Task.Run(()=>session.Run(inputs));
        float[] probs = results.First(x => x.Name == "prediction").AsEnumerable<float>().ToArray();
        var probs_with_label = probs.Zip(outputMap).OrderByDescending(x => x.First).ToList();
        return (
            probs_with_label.Where(x => tagMeta[x.Second].category == 0 && x.First > tagMeta[x.Second].best_threshold).Take(50).Select(p => new TagInfo(p.Second,p.First)).ToList(),
            probs_with_label.Where(x => tagMeta[x.Second].category == 4 && x.First > tagMeta[x.Second].best_threshold).Take(30).Select(p => new TagInfo(p.Second, p.First)).ToList(),
            probs_with_label.Where(x => tagMeta[x.Second].category == 9).Select(p => new TagInfo(p.Second, p.First)).ToList()
        );
    }
}
[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(TimmConfig))]
internal partial class SourceGenerationContext : JsonSerializerContext
{
}