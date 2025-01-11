using IAUN.InformationRetrieval.Final;
using System.Diagnostics;

Console.WriteLine("Hello, Welcome to my first search engine!");
const string sourcePath = "E:\\PHD\\Codes_Practices\\Information retrieval\\Final\\-1873142016_1023851046\\Information-Retrieval-on-CISI-master\\CISI.ALL";
const string booleanQueries = "E:\\PHD\\Codes_Practices\\Information retrieval\\Final\\-1873142016_1023851046\\Information-Retrieval-on-CISI-master\\CISI.BLN";
const string relPath = "E:\\PHD\\Codes_Practices\\Information retrieval\\Final\\-1873142016_1023851046\\Information-Retrieval-on-CISI-master\\CISI.REL";
const string onnx = "E:\\PHD\\Codes_Practices\\Information retrieval\\Final\\-1873142016_1023851046\\Bert\\model_O1.onnx";
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

parser.Quires.ForEach(async query =>
{

	Console.WriteLine($"Query #{++queryIndex}");

	var condition = parser.Parse(query);

	var sw = new Stopwatch();
	sw.Start();
	var result = await documents.Search(condition);
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


Console.WriteLine($"{new string('-',20)} BERT MODEL {new string('-', 20)}");

#region BERT
var bertProcessor = new BertEmbeddings(sourcePath, relPath, vocab, database, onnx);
await bertProcessor.PrepareDocuments();
var output=bertProcessor.Search("Image recognition and any other methods of automatically transforming printed text into computer-ready form.", 4);
foreach (var item in output)
{
	Console.WriteLine($"{item.DocumentId} ---- {item.Similarity}");
}
#endregion

Console.WriteLine("Good luck!");