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
using System.Formats.Tar;
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
    public IStorageFile? modelFile;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(Image))]
    public IStorageFile? imageFile;

    [ObservableProperty]
    public IStorageFile? tagsFile;

    [ObservableProperty]
    public IStorageFile? configFile;

    [ObservableProperty]
    public partial List<TagInfo>? GeneralTags { get; set; }

    [ObservableProperty]
    public partial List<TagInfo>? CharacterTags { get; set; }

    [ObservableProperty]
    public partial List<TagInfo>? RatingTags { get; set; }

    public IStorageProvider StorageProvider { get; init; } = provider;

    public Task<Bitmap?> Image => ImageFile is null ? Task.FromResult<Bitmap?>(null) : ImageHelper.LoadFromFileAsync(ImageFile);

    [RelayCommand]
    public async Task InferenceAsync()
    {
        if (ModelFile is null || ImageFile is null || TagsFile is null || ConfigFile is null) return;

        await using var modelStream = await ModelFile.OpenReadAsync();
        await using var imageStream = await ImageFile.OpenReadAsync();
        await using var tagsStream = await TagsFile.OpenReadAsync();
        await using var configStream = await ConfigFile.OpenReadAsync();

        var result = await CaformerDbv4Inference.InferenceAsync(modelStream, imageStream, tagsStream, configStream);
        (GeneralTags, CharacterTags, RatingTags) = result;
    }

    [RelayCommand]
    public async Task SelectModelPathAsync()
    {
        var file = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions { AllowMultiple = false, SuggestedFileName = "model.onnx" });
        if (file.Count is 0) return;
        ModelFile = file[0];
    }


    [RelayCommand]
    public async Task SelectImagePathAsync()
    {
        var file = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions { AllowMultiple = false });
        if (file.Count is 0) return;
        ImageFile = file[0];
    }

    [RelayCommand]
    public async Task SelectTagsPathAsync()
    {
        var file = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions { AllowMultiple = false, SuggestedFileName = "selected_tags.csv" });
        if (file.Count is 0) return;
        TagsFile = file[0];
    }

    [RelayCommand]
    public async Task SelectConfigPathAsync()
    {
        var file = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions { AllowMultiple = false, SuggestedFileName = "config.json" });
        if (file.Count is 0) return;
        ConfigFile = file[0];
    }
}
public static class ImageHelper
{
    public static async Task<Bitmap> LoadFromFileAsync(IStorageFile file)
    {
        await using var stream = await file.OpenReadAsync();
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

    static async Task<DenseTensor<float>> PreprocessAsync(Stream imageStream)
    {
        // 将输入流复制到内存，避免非可寻址流在异步场景下的问题
        using var ms = new MemoryStream();
        await imageStream.CopyToAsync(ms).ConfigureAwait(false);
        var imageBytes = ms.ToArray();

        // 将重 CPU/内存操作放到后台线程，避免阻塞 UI（安卓上可避免 ANR 闪退）
        return await Task.Run(() =>
        {
            using var image = Image.Load<Rgba32>(imageBytes);

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
                    tensor[0, 0, y, x] = (r - Mean[0]) / Std[0];
                    tensor[0, 1, y, x] = (g - Mean[1]) / Std[1];
                    tensor[0, 2, y, x] = (b - Mean[2]) / Std[2];
                }
            }
            return tensor;
        }).ConfigureAwait(false);
    }

    static List<TagRecord> LoadTagMeta(Stream csvStream)
    {
        using var reader = new StreamReader(csvStream);
        using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
        return csv.GetRecords<TagRecord>().ToList();
    }

    static void ShowResult(List<(float, string)> general_tags)
    {
        foreach (var (prob, label) in general_tags)
            Console.WriteLine($"  - {label} ({prob:P2})");
    }

    public static async Task<(List<TagInfo> general_tags, List<TagInfo> character_tags, List<TagInfo> rating_tags)>
        InferenceAsync(Stream modelStream, Stream imageStream, Stream tagsCsvStream, Stream configStream)
    {
        var inputTensor = await PreprocessAsync(imageStream);
        var tagMeta = LoadTagMeta(tagsCsvStream).ToDictionary(x => x.name);
        var timmConfig = await JsonSerializer.DeserializeAsync<TimmConfig>(configStream, new JsonSerializerOptions { TypeInfoResolver = SourceGenerationContext.Default });
        var outputMap = timmConfig!.tags;

        using var memoryStream = new MemoryStream();
        await modelStream.CopyToAsync(memoryStream);
        var modelBytes = memoryStream.ToArray();

        using var session = new InferenceSession(modelBytes);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };
        using var results = await Task.Run(() => session.Run(inputs));
        float[] probs = results.First(x => x.Name == "prediction").AsEnumerable<float>().ToArray();
        var probs_with_label = probs.Zip(outputMap).OrderByDescending(x => x.First).ToList();
        return (
            probs_with_label.Where(x => tagMeta[x.Second].category == 0 && x.First > tagMeta[x.Second].best_threshold).Take(50).Select(p => new TagInfo(p.Second, p.First)).ToList(),
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