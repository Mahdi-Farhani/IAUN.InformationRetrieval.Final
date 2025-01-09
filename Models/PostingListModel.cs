namespace IAUN.InformationRetrieval.Final.Models;

public class PostingListModel
{
	public int Count => DocumentIdWithFrequency.Sum(x => x.Frequency);
	public List<PostingListInfo> DocumentIdWithFrequency { get; set; } = [];
}
