namespace IAUN.InformationRetrieval.Final.Models;

public class Document
{
	public int Id { get; set; }
	public string Title { get; set; }=string.Empty;
	public string Author { get; set; }=string.Empty;
	public string Content { get; set; }=string.Empty;
	public string ExtraData { get; set; } = string.Empty;
}
