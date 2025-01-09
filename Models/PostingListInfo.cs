namespace IAUN.InformationRetrieval.Final.Models;

public record PostingListInfo
{
	public PostingListInfo(int documentId,int frequency)
	{
		DocumentId= documentId;
		Frequency= frequency;
	}
	public int DocumentId { get; set; }
	public int Frequency { get; set; }
}
