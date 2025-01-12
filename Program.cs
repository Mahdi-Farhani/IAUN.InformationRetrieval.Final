using IAUN.InformationRetrieval.Final;
using IAUN.InformationRetrieval.Final.Models;
using System.Diagnostics;
using static Lucene.Net.Util.Fst.Util;

Console.WriteLine("Hello, Welcome to my first search engine!");
const string sourcePath = "E:\\PHD\\Codes_Practices\\Information retrieval\\Final\\-1873142016_1023851046\\Information-Retrieval-on-CISI-master\\CISI.ALL";
const string queryPath = "E:\\PHD\\Codes_Practices\\Information retrieval\\Final\\-1873142016_1023851046\\Information-Retrieval-on-CISI-master\\CISI.QRY";
const string booleanQueries = "E:\\PHD\\Codes_Practices\\Information retrieval\\Final\\-1873142016_1023851046\\Information-Retrieval-on-CISI-master\\CISI.BLN";
const string relPath = "E:\\PHD\\Codes_Practices\\Information retrieval\\Final\\-1873142016_1023851046\\Information-Retrieval-on-CISI-master\\CISI.REL";
const string onnx = "E:\\PHD\\Codes_Practices\\Information retrieval\\Final\\-1873142016_1023851046\\Bert\\model.onnx";
const string vocab = "E:\\PHD\\Codes_Practices\\Information retrieval\\Final\\-1873142016_1023851046\\Bert\\vocab.txt";

var database = Database.GetDatabase();

#region Boolean Query
var documents = new DocumentParser(sourcePath, relPath, database);
var forceToInitialize = false;
await documents.FetchAllDocumentsAsync(forceToInitialize);
await documents.TokenizingAsync(forceToInitialize);
await documents.CreateInvertedIndexAsync(forceToInitialize);
await documents.LoadRelFile();

var parser = new BooleanQueryParser();
await parser.LoadFile(booleanQueries);
var queryIndex = 0;
Console.WriteLine(new string('-', 20));
var allRetrieved = new List<List<PostingListInfo>>();
parser.Quires.ForEach(async query =>
{

	Console.WriteLine($"Query #{++queryIndex}");

	var condition = parser.Parse(query);

	var sw = new Stopwatch();
	sw.Start();
	var result = await documents.Search(condition);
	allRetrieved.Add(result ?? []);

	var founded = result?.Select(x => x.DocumentId.ToString()).Aggregate((x, c) => x + "," + c);
	Console.WriteLine(founded);

	var precision = documents.Precision(result, queryIndex);
	Console.WriteLine($"Precision for query #{queryIndex} = {precision:F4}");

	var recall = documents.ReCall(result, queryIndex);
	Console.WriteLine($"Re-Call for query #{queryIndex} = {recall:F4}");

	var fMeasure = documents.FMeasure(result, queryIndex);
	Console.WriteLine($"F-Measure for query #{queryIndex} = {fMeasure:F4}");

	sw.Stop();
	Console.WriteLine($"result in {sw.ElapsedMilliseconds} ms.");
	Console.WriteLine(new string('-', 20));
});

#endregion
var map = documents.MAP(allRetrieved!, queryIndex);
Console.WriteLine($"MAP for all queries = {map:F4}");


Console.WriteLine($"{new string('-', 20)} BERT MODEL {new string('-', 20)}");

#region BERT
var bertProcessor = new BertEmbeddings(sourcePath, relPath, vocab, onnx, queryPath, database);
await bertProcessor.PrepareDocuments();
var index = 0;
var allBertRetrieved = new List<List<DocumentSimilarity>>();

bertProcessor.Queries.ForEach(query =>
{
	var output = bertProcessor.Search(query, index++);
	allBertRetrieved.Add(output);
	
	foreach (var item in output.Take(10).ToList())
	{
		Console.WriteLine($"{item.DocumentId} ---- {item.Similarity}");
	}
	var precision = bertProcessor.Precision(output, queryIndex);
	Console.WriteLine($"Precision for query #{queryIndex} = {precision:F4}");

	var recall = bertProcessor.ReCall(output, queryIndex);
	Console.WriteLine($"Re-Call for query #{queryIndex} = {recall:F4}");

	var fMeasure = bertProcessor.FMeasure(output, queryIndex);
	Console.WriteLine($"F-Measure for query #{queryIndex} = {fMeasure:F4}");

	Console.WriteLine(new string('-', 20));
});
map = bertProcessor.MAP(allBertRetrieved!, queryIndex);
Console.WriteLine($"MAP for all queries = {map:F4}");
Console.WriteLine(new string('-', 20));

#endregion

Console.WriteLine("Good luck!");