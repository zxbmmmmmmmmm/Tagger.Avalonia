using Avalonia.Controls;
using Tagger.Avalonia.ViewModels;

namespace Tagger.Avalonia.Views;

public partial class MainView : UserControl
{
    public MainView()
    {
        InitializeComponent();
        this.Loaded += (s, e) =>
        {
            this.DataContext = new MainViewModel(TopLevel.GetTopLevel(this)!.StorageProvider);
        };
    }
}