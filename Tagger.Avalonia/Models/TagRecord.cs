namespace Tagger.Avalonia.Models;

class TagRecord
{
    public string name { get; set; } = string.Empty;
    public int category { get; set; }
    public float best_threshold { get; set; }
}