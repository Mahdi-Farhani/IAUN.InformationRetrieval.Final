using IAUN.InformationRetrieval.Final.Models;
using Lucene.Net.Analysis;
using Lucene.Net.Analysis.Core;
using Lucene.Net.Analysis.En;
using Lucene.Net.Analysis.Standard;
using Lucene.Net.Util;
using StackExchange.Redis;
using System.Diagnostics;
using System.Text.RegularExpressions;

namespace IAUN.InformationRetrieval.Final;

public partial class DocumentParser(string sourcePath,string relPath, IDatabase database)
{
	private const string DocumentKey = "Documents";
	private const string TokenKey = "DocumentTokens";
	private const string InvertedIndexTokenKey = "InvertedIndex";

	private const LuceneVersion CurrentLuceneVersion = LuceneVersion.LUCENE_48;

	private readonly string sourcePath = sourcePath;
	private readonly string relPath = relPath;
	private readonly IDatabase database = database;

	public List<Document> Documents { get; set; } = [];
	public List<TokenModel> DocumentsTokens { get; set; } = [];
	public List<ResultModel> Result { get; set; } = [];
	public Dictionary<string, PostingListModel> InvertedIndex { get; set; } = [];

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
	public async Task TokenizingAsync(bool force = false)
	{
		var sw = new Stopwatch();
		sw.Start();
		if (Documents.Count == 0)
		{
			Console.WriteLine("Documents list is empty");
			return;
		}
		if (DocumentsTokens.Count > 0 && !force)
		{
			Console.WriteLine("documents already tokenized");
			return;
		}
		if (database.KeyExists(TokenKey))
		{
			var tokenStringData = await database.StringGetAsync(TokenKey);
			if (!tokenStringData.IsNull && !force)
			{
				DocumentsTokens = System.Text.Json.JsonSerializer.Deserialize<List<TokenModel>>(tokenStringData.ToString()) ?? [];
				Console.WriteLine($"{DocumentsTokens.Count} documents tokens fetched from cache");
				return;
			}
		}


		using var analyzer = GetAnalyzer();
		Documents.ForEach((document) =>
		{

			var documentToken = new TokenModel { DocumentId = document.Id };
			using (var contentTokenStream = analyzer.GetTokenStream("content", document.Content))
			{
				contentTokenStream.Reset();
				while (contentTokenStream.IncrementToken())
				{
					var term = contentTokenStream.GetAttribute<Lucene.Net.Analysis.TokenAttributes.ICharTermAttribute>();
					documentToken.ContentTokens.Add(term.ToString());
				}
				contentTokenStream.End();
			}

			using (var titleTokenStream = analyzer.GetTokenStream("title", document.Title))
			{

				titleTokenStream.Reset();
				while (titleTokenStream.IncrementToken())
				{
					var term = titleTokenStream.GetAttribute<Lucene.Net.Analysis.TokenAttributes.ICharTermAttribute>();
					documentToken.TitleTokens.Add(term.ToString());
				}

				titleTokenStream.End();
			}
			DocumentsTokens.Add(documentToken);


		});


		var tokenData = System.Text.Json.JsonSerializer.Serialize(DocumentsTokens);
		await database.StringGetSetAsync(TokenKey, tokenData!);

		sw.Stop();
		Console.WriteLine($"tokenizing data took {sw.ElapsedMilliseconds} ms");
	}

	private static Analyzer GetAnalyzer()
	{
		return Analyzer.NewAnonymous((fieldName, reader) =>
		{
			var tokenizer = new StandardTokenizer(CurrentLuceneVersion, reader);
			var lowerCaseFilter = new LowerCaseFilter(CurrentLuceneVersion, tokenizer);
			var stopWords = new StopFilter(CurrentLuceneVersion, lowerCaseFilter, StopAnalyzer.ENGLISH_STOP_WORDS_SET);
			var steamingFilter = new PorterStemFilter(stopWords);
			var finalStream = new TokenStreamComponents(tokenizer, steamingFilter);
			return finalStream;
		});
	}


	public async Task CreateInvertedIndexAsync(bool force = false)
	{
		var sw = new Stopwatch();
		sw.Start();
		if (Documents.Count == 0)
		{
			Console.WriteLine("Documents list is empty");
			return;
		}
		if (InvertedIndex.Count > 0 && !force)
		{
			Console.WriteLine("inverted index already created");
			return;
		}

		if (database.KeyExists(InvertedIndexTokenKey))
		{
			var invertedTokenStringData = await database.StringGetAsync(InvertedIndexTokenKey);
			if (!invertedTokenStringData.IsNull && !force)
			{
				InvertedIndex = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, PostingListModel>>(invertedTokenStringData.ToString()) ?? [];
				Console.WriteLine($"{InvertedIndex.Count} invertedIndex fetched from cache");
				return;
			}
		}

		DocumentsTokens.ForEach(token =>
		{
			token.ContentTokens.ForEach(contentToken =>
			{
				InvertedIndexAddOrUpdate(token.DocumentId, contentToken);
			});
			token.TitleTokens.ForEach(titleToken =>
			{
				InvertedIndexAddOrUpdate(token.DocumentId, titleToken);
			});
		});

		var invertedIndexData = System.Text.Json.JsonSerializer.Serialize(InvertedIndex);
		await database.StringGetSetAsync(InvertedIndexTokenKey, invertedIndexData!);


		sw.Stop();
		Console.WriteLine($"fetching data took {sw.ElapsedMilliseconds} ms");

	}
	public void ShowInvertedIndex(int start = 0, int length = 10)
	{
		Console.WriteLine($"Total Index :{InvertedIndex.Count}");
		InvertedIndex.Skip((start * length) + 1).Take(length).ToList().ForEach(index =>
		{
			Console.WriteLine($"{index.Key}\t{index.Value.Count}\t" +
				$"{index.Value.DocumentIdWithFrequency.Select(z => z.DocumentId.ToString()).Aggregate((x, c) => x + "," + c)}");
		});
		Console.WriteLine(new string('-', 50));
	}
	public void SaveInvertedIndex()
	{
		using var file = new StreamWriter("invertedIndex.csv");
		InvertedIndex.ToList().ForEach(index =>
		{

			file.WriteLine($"{index.Key},{index.Value.Count}," +
				$"{index.Value.DocumentIdWithFrequency.Select(z => z.DocumentId.ToString()).Aggregate((x, c) => x + "-" + c)}");
		});

	}
	private void InvertedIndexAddOrUpdate(int documentId, string term)
	{
		var postingList = new PostingListModel();

		var hasKey = InvertedIndex.TryGetValue(term, out postingList);
		postingList = postingList ?? new PostingListModel();
		if (hasKey)
		{
			var previousPostingList = postingList.DocumentIdWithFrequency.SingleOrDefault(x => x.DocumentId == documentId);
			if (previousPostingList == null)
			{
				postingList.DocumentIdWithFrequency.Add(new PostingListInfo(documentId, 1));
			}
			else
			{
				previousPostingList.Frequency += 1;
			}
		}
		else
		{
			postingList.DocumentIdWithFrequency.Add(new PostingListInfo(documentId, 1));
			InvertedIndex.Add(term, postingList);
		}
	}

	public async Task<List<PostingListInfo>?> Search(QueryNode condition)
	{
		if (condition is TermNode termNode)
		{
			PostingListModel? value = null;
			using var analyzer = GetAnalyzer();
			using (var contentTokenStream = analyzer.GetTokenStream("term", termNode.Term))
			{
				contentTokenStream.Reset();
				var tokenized=contentTokenStream.IncrementToken();
				if (tokenized)
				{
					var term = contentTokenStream.GetAttribute<Lucene.Net.Analysis.TokenAttributes.ICharTermAttribute>();
					InvertedIndex.TryGetValue(term.ToString(), out value);
				}
				contentTokenStream.End();
			}

			return value?.DocumentIdWithFrequency;
		}
		if (condition is LogicalNode logicalNode)
		{
			var postingListToFilter = new List<List<PostingListInfo>>();
			foreach (var conditionItem in logicalNode.Operands)
			{
				var postingList = await Search(conditionItem);
				if (postingList == null) continue;
				postingListToFilter.Add(postingList);
			}
			return logicalNode.Operator switch
			{
				Operator.And => AndConditionPostingList(postingListToFilter),
				Operator.Or => OrConditionPostingList(postingListToFilter),
				_ => null,
			};
		}
		return null;

	}

	private static List<PostingListInfo>? OrConditionPostingList(List<List<PostingListInfo>> postingListToFilter)
	{
		var answer=new List<PostingListInfo>();
		foreach (var item in postingListToFilter)
		{
			answer.AddRange(item);
		}
		return answer;	
	}

	private static List<PostingListInfo>? AndConditionPostingList(List<List<PostingListInfo>> postingListToFilter)
	{
		if (postingListToFilter == null || postingListToFilter.Count==0) return null;
		var finalAnswer=postingListToFilter[0];
		for (int i = 1; i < postingListToFilter.Count; i++)
		{
			var secondPostingList = postingListToFilter[i];
			finalAnswer= Merge(finalAnswer, secondPostingList);
		}
		return finalAnswer;
		

	}

	private static List<PostingListInfo> Merge(List<PostingListInfo> finalAnswer, List<PostingListInfo> secondPostingList)
	{
		var i = 0;
		var j = 0;
		var answer = new List<PostingListInfo>();
		while( i< finalAnswer.Count && j<secondPostingList.Count)
		{
			if (finalAnswer[i].DocumentId == secondPostingList[j].DocumentId)
			{
				answer.Add(finalAnswer[i]);
				i++;
				j++;
			}
			else if (finalAnswer[i].DocumentId < secondPostingList[j].DocumentId)
			{
				i++;
			}
			else
			{
				j++;
			}
		}
		return answer;
	}
	public async Task LoadRelFile()
	{
		var sw = new Stopwatch();
		sw.Start();
		var allResultText = await File.ReadAllTextAsync(relPath, System.Text.Encoding.UTF8);
		var allResult = EvaluationSplitter().Matches(allResultText).ToList();
		foreach (var item in allResult)
		{
			Result.Add(new ResultModel {  QueryNumber=int.Parse(item.Groups[1].Value), DocumentId = int.Parse(item.Groups[2].Value) });
		}

		sw.Stop();
		Console.WriteLine($"Query file fetched in {sw.ElapsedMilliseconds} ms.");
	}
	public decimal Precision(List<PostingListInfo>? result, int queryNumber)
	{
		var relevantDocuments = Result.Where(x=>x.QueryNumber==queryNumber).ToList();
		var relevantDocIds=relevantDocuments.Select(x=>x.DocumentId).ToList();
		var relevantRetrieved= result?.Where(x => relevantDocIds.Contains(x.DocumentId)).Count()??0;
		
		return relevantRetrieved/(result?.Count ?? 0m);
	}
	public decimal ReCall(List<PostingListInfo>? result,int queryNumber)
	{
		var relevantDocuments = Result.Where(x=>x.QueryNumber==queryNumber).ToList();
		var relevantDocIds = relevantDocuments.Select(x => x.DocumentId).ToList();
		var relevantRetrieved = result?.Where(x => relevantDocIds.Contains(x.DocumentId)).Count() ?? 0;

		return relevantRetrieved / relevantDocuments.Count;
	}

	public decimal FMeasure(List<PostingListInfo>? result, int queryNumber)
	{
		var precision=Precision(result,queryNumber);
		var recall= ReCall(result,queryNumber);
		if (precision == 0 && recall == 0) return 0;
		return ((precision*recall) / (precision+recall))*2;
	}


	[GeneratedRegex(@"^\n?\.I\s+\d+", RegexOptions.Multiline)]
	private static partial Regex DocumentSplitter();

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
