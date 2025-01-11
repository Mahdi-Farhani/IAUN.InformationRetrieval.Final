namespace IAUN.InformationRetrieval.Final.Models;

public class IndexModel
{
	public int Item1 { get; set; }
	public int Item2 { get; set; }
	public int Item3 { get; set; }

}

public class ResultModel
{
	public int QueryNumber { get; set; }
	public int DocumentId { get; set; }

}

public class DocumentEmbedding
{
	public int DocumentId { get; set; }
	public float[] Embeddings { get; set; } = [];
}
public class DocumentSimilarity
{
	public int DocumentId { get; set; }
	public int QueryId { get; set; }
	public float Similarity { get; set; }
}