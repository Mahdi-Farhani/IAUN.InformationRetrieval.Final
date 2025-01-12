using IAUN.InformationRetrieval.Final.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using StackExchange.Redis;
using System.Diagnostics;
using System.Text.RegularExpressions;

namespace IAUN.InformationRetrieval.Final;
public partial class BertEmbeddings(string sourcePath, string relPath, string vocabPath, string modelPath, string queryPath, IDatabase database)
{
	private const string DocumentKey = "Documents";
	private const string TokenKey = "Bert_DocumentTokens";

	private readonly string sourcePath = sourcePath;
	private readonly string relPath = relPath;
	private readonly string vocabPath = vocabPath;
	private readonly string queryPath = queryPath;
	private readonly IDatabase database = database;
	private Dictionary<string, int> vocab = [];
	private readonly InferenceSession session = new(modelPath);
	public List<Document> Documents { get; set; } = [];
	public List<string> Queries { get; set; } = [];
	public List<TokenModel> DocumentsTokens { get; set; } = [];
	public List<ResultModel> Result { get; set; } = [];
	public List<DocumentEmbedding> DocumentEmbeddings { get; set; } = [];
	public List<DocumentSimilarity> DocumentSimilarities { get; set; } = [];
	public async Task<int> FetchAllDocumentsAsync(bool force = false)
	{

		if (database.KeyExists(DocumentKey) && !force)
		{
			var documentsStringData = await database.StringGetAsync(DocumentKey);
			if (!documentsStringData.IsNull)
			{
				Documents = System.Text.Json.JsonSerializer.Deserialize<List<Document>>(documentsStringData.ToString()) ?? [];
				Console.WriteLine($"{Documents.Count} documents fetched from cache");
				return Documents.Count;
			}
		}
		await ReadDocumentsAsync();
		return Documents.Count;
	}
	public List<string> LoadQueries()
	{
		throw new NotImplementedException();
	}
	private async Task ReadDocumentsAsync()
	{
		var sw = new Stopwatch();
		sw.Start();
		var allDocumentsText = await File.ReadAllTextAsync(sourcePath, System.Text.Encoding.UTF8);

		var allDocuments = DocumentSplitter().Split(allDocumentsText).ToList();
		allDocuments.RemoveAll(string.IsNullOrEmpty);
		Console.WriteLine($"{allDocuments.Count} find");
		var index = 0;
		foreach (var document in allDocuments)
		{
			var documentId = ++index;
			var documentTitle = string.Empty;
			var documentAuthor = string.Empty;
			var documentContent = string.Empty;
			var documentExtra = string.Empty;


			var documentTitleMatch = DocumentTitle().Match(document);
			if (documentTitleMatch.Success)
			{
				documentTitle = documentTitleMatch.Groups[1].Value.Trim();
			}

			var documentAuthorMatch = DocumentAuthor().Match(document);
			if (documentAuthorMatch.Success)
			{
				documentAuthor = documentAuthorMatch.Groups[1].Value.Trim();
			}
			var documentContentMatch = DocumentContent().Match(document);
			if (documentContentMatch.Success)
			{
				documentContent = documentContentMatch.Value.Trim();
			}
			var documentExtraMatch = DocumentExtra().Match(document);
			if (documentExtraMatch.Success)
			{
				documentExtra = documentExtraMatch.Groups[1].Value.Trim();
			}

			Documents.Add(new Document
			{
				Author = documentAuthor,
				Content = documentContent,
				ExtraData = documentExtra,
				Id = documentId,
				Title = documentTitle,
			});



		}
		var documentsData = System.Text.Json.JsonSerializer.Serialize(Documents);
		await database.StringGetSetAsync(DocumentKey, documentsData!);
		sw.Stop();
		Console.WriteLine($"fetching data took {sw.ElapsedMilliseconds} ms");
	}
	private async Task FetchAllQueriesAsync()
	{
		var sw = new Stopwatch();
		sw.Start();
		var allQueriesText = await File.ReadAllTextAsync(queryPath, System.Text.Encoding.UTF8);

		var allQueries = QuerySplitter().Matches(allQueriesText).ToList();
		Console.WriteLine($"{allQueries.Count} queries found");
		foreach (var item in allQueries)
		{

			var query = item.Value.TrimStart().TrimEnd();
			Queries.Add(query);


		}
		sw.Stop();
		Console.WriteLine($"queries data took {sw.ElapsedMilliseconds} ms");
	}

	public async Task PrepareDocuments()
	{
		var sw = new Stopwatch();
		sw.Start();
		Console.WriteLine("Start preparing documents");
		var loadVocab = LoadVocab();
		var loadQueries = FetchAllQueriesAsync();
		var loadDocuments = FetchAllDocumentsAsync();
		var loadRel = LoadRelFile();
		await Task.WhenAll(loadQueries, loadDocuments, loadVocab,loadRel);

		foreach (var document in Documents)
		{
			var embeddings = GetEmbeddings(document.Content);
			DocumentEmbeddings.Add(new DocumentEmbedding { DocumentId = document.Id, Embeddings = embeddings });
		}
		sw.Stop();
		Console.WriteLine($"preparing documents completed in {sw.ElapsedMilliseconds} ms");
	}
	public List<DocumentSimilarity> Search(string query, int queryId)
	{
		var queryEmbeddings = GetEmbeddings(query);

		foreach (var item in DocumentEmbeddings)
		{
			var similarity = Similarity.CosineSimilarity(queryEmbeddings, item.Embeddings);
			DocumentSimilarities.Add(new DocumentSimilarity { DocumentId = item.DocumentId, QueryId = queryId, Similarity = similarity });

		}
		return [.. DocumentSimilarities.Where(x => x.QueryId == queryId).OrderByDescending(x => x.Similarity)];
	}

	public static List<string> Tokenizing(string text)
	{
		var sw = new Stopwatch();
		sw.Start();
		text = text.ToLower().Trim();
		var output = text.Split(' ').ToList();
		sw.Stop();
		Console.WriteLine($"tokenizing query took {sw.ElapsedMilliseconds} ms");
		return output;
	}
	private static Tensor<long> GenerateTokenTypeIds(int inputIdsLength)
	{
		var tokenTypeIds = new DenseTensor<long>([1, inputIdsLength]);
		return tokenTypeIds;
	}
	private static Tensor<long> GenerateAttentionMask(long[] inputIds)
	{
		var mask = inputIds.Select(id => id > 0 ? 1 : 0).ToArray();
		var tensor = new DenseTensor<long>([1, inputIds.Length]);
		for (int i = 0; i < mask.Length; i++)
		{
			tensor[0, i] = mask[i];
		}
		return tensor;
	}
	public float[] GetEmbeddings(string text)
	{
		var inputTensor = PreProcessText(text);
		var tokenTypeIds = GenerateTokenTypeIds(inputTensor.Count());
		var attentionMask = GenerateAttentionMask([.. inputTensor]);

		var inputs = new List<NamedOnnxValue> {
			NamedOnnxValue.CreateFromTensor("input_ids",inputTensor),
			NamedOnnxValue.CreateFromTensor("token_type_ids",tokenTypeIds),
			NamedOnnxValue.CreateFromTensor("attention_mask",attentionMask)
		};
		using var results = session.Run(inputs);
		var output = results.FirstOrDefault(x => x.Name == "last_hidden_state")?.AsTensor<float>();
		return output?.ToArray() ?? [];

	}
	public async Task LoadVocab()
	{
		var sw = new Stopwatch();
		sw.Start();
		var allResultText = await File.ReadAllTextAsync(vocabPath, System.Text.Encoding.UTF8);
		var allResult = allResultText.Split('\n').ToList();


		for (int i = 0; i < allResult.Count; i++)
		{
			vocab.TryAdd(allResult[i], i);
		}
		sw.Stop();
		Console.WriteLine($"vocab file fetched in {sw.ElapsedMilliseconds} ms.");
	}
	public async Task LoadRelFile()
	{
		var sw = new Stopwatch();
		sw.Start();
		var allResultText = await File.ReadAllTextAsync(relPath, System.Text.Encoding.UTF8);
		var allResult = EvaluationSplitter().Matches(allResultText).ToList();
		foreach (var item in allResult)
		{
			Result.Add(new ResultModel { QueryNumber = int.Parse(item.Groups[1].Value), DocumentId = int.Parse(item.Groups[2].Value) });
		}

		sw.Stop();
		Console.WriteLine($"Query file fetched in {sw.ElapsedMilliseconds} ms.");
	}
	public decimal Precision(List<DocumentSimilarity>? result, int queryNumber)
	{
		var relevantDocuments = Result.Where(x => x.QueryNumber == queryNumber).ToList();
		var relevantDocIds = relevantDocuments.Select(x => x.DocumentId).ToList();
		var relevantRetrieved = result?.Where(x => relevantDocIds.Contains(x.DocumentId)).Count() ?? 0;

		return relevantRetrieved / (result?.Count ?? 0m);
	}
	public decimal ReCall(List<DocumentSimilarity>? result, int queryNumber)
	{
		var relevantDocuments = Result.Where(x => x.QueryNumber == queryNumber).ToList();
		var relevantDocIds = relevantDocuments.Select(x => x.DocumentId).ToList();
		var relevantRetrieved = result?.Where(x => relevantDocIds.Contains(x.DocumentId)).Count() ?? 0;
		if (relevantDocuments.Count == 0) return 0;
		return relevantRetrieved / relevantDocuments.Count;
	}

	public decimal FMeasure(List<DocumentSimilarity>? result, int queryNumber)
	{
		var precision = Precision(result, queryNumber);
		var recall = ReCall(result, queryNumber);
		if (precision == 0 && recall == 0) return 0;
		return ((precision * recall) / (precision + recall)) * 2;
	}
	private decimal CalculateAveragePrecision(List<DocumentSimilarity> retrievedDocs, int queryNumber)
	{
		var averagePrecision = 0m;
		int relevantRetrieved = 0;

		for (int i = 0; i < retrievedDocs.Count; i++)
		{
			if (Result.Where(x => x.QueryNumber == queryNumber).Any(x => x.DocumentId == retrievedDocs[i].DocumentId))
			{
				relevantRetrieved++;
				var precisionAtK = (decimal)relevantRetrieved / (i + 1);
				averagePrecision += precisionAtK;
			}
		}

		return Result.Count > 0 ? averagePrecision / Result.Count : 0m;
	}


	public decimal MAP(List<List<DocumentSimilarity>> result)
	{
		var totalAP = 0m;

		for (int i = 0; i < result.Count; i++)
		{
			var ap = CalculateAveragePrecision(result[i], i);
			totalAP += ap;
		}

		return totalAP / result.Count;
	}
	private Tensor<long> PreProcessText(string text)
	{
		text = text.Replace("\r", " ").Replace("\n", " ");
		var tokens = Tokenizing(text);
		var inputIds = new List<long>();
		tokens.Insert(0, "[CLS]");
		tokens.Add("[SEP]");

		foreach (var token in tokens.Where(x => !string.IsNullOrWhiteSpace(x)).ToList())
		{
			var exists = vocab.TryGetValue(token, out int id);

			inputIds.Add(exists ? id : vocab["[UNK]"]);
		}
		int maxSequenceLength = 256;
		if (inputIds.Count > maxSequenceLength)
		{
			inputIds = inputIds.Take(maxSequenceLength).ToList();
		}
		else
		{
			while (inputIds.Count < maxSequenceLength)
			{
				inputIds.Add(0);
			}
		}

		var inputTensor = new DenseTensor<long>([1, maxSequenceLength]);
		for (int i = 0; i < maxSequenceLength; i++)
		{
			inputTensor[0, i] = inputIds[i];
		}

		return inputTensor;
	}

	[GeneratedRegex(@"^\n?\.I\s+\d+", RegexOptions.Multiline)]
	private static partial Regex DocumentSplitter();
	[GeneratedRegex(@"(?<=^\.W\s)([\s\S]*?)(?=\n\.\w)", RegexOptions.Multiline | RegexOptions.Singleline)]
	private static partial Regex QuerySplitter();
	[GeneratedRegex("\\n\\.T\\s(.+)")]
	private static partial Regex DocumentTitle();
	[GeneratedRegex("\\n\\.A\\s(.+)")]
	private static partial Regex DocumentAuthor();
	[GeneratedRegex("(?<=\\.W\\s).*(?=\\n\\.X)", RegexOptions.Singleline)]
	private static partial Regex DocumentContent();	
	[GeneratedRegex("\\n\\.X\\s(.+)")]
	private static partial Regex DocumentExtra();
	[GeneratedRegex("(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+\\.\\d+)")]
	private static partial Regex EvaluationSplitter();


}
