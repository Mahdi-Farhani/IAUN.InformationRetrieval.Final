namespace IAUN.InformationRetrieval.Final.Models;

public class TokenModel
{
	public int DocumentId { get; set; }
	public List<string> ContentTokens { get; set; } = [];
	public List<string> TitleTokens { get; set; } = [];

}