using IAUN.InformationRetrieval.Final.Models;
using System.Diagnostics;
using System.Text.RegularExpressions;

namespace IAUN.InformationRetrieval.Final;

public partial class BooleanQueryParser
{
	public List<string> Quires { get; set; } = [];

	private string query = string.Empty;
	private int position;
	

	public async Task LoadFile(string sourcePath)
	{
		var sw = new Stopwatch();
		sw.Start();
		var allQueriesText= await File.ReadAllTextAsync(sourcePath, System.Text.Encoding.UTF8);
		var allQueries = QuerySplitter().Matches(allQueriesText).ToList();
		foreach (var item in allQueries)
		{
			Quires.Add(item.Groups[1].Value);
		}

		sw.Stop();
		Console.WriteLine($"Query file fetched in {sw.ElapsedMilliseconds} ms.");
	}
	public QueryNode Parse(string query)
	{
		this.query = query.Replace("\n", "").Replace("\r", "");
		position = 0;
		return ParseExpression();
	}

	private QueryNode ParseExpression()
	{
		SkipWhitespace();

		if (query[position] == '#')
		{
			position++; // Skip '#'
			Operator operatorName = ParseOperator();

			var logicalNode = new LogicalNode { Operator = operatorName };
			Expect('(');

			while (query[position] != ')')
			{
				logicalNode.Operands.Add(ParseExpression());
				SkipWhitespace();

				if (query[position] == ',')
					position++; // Skip ','
			}

			Expect(')');
			return logicalNode;
		}
		else if (query[position] == '\'')
		{
			return new TermNode { Term = ParseTerm() };
		}

		throw new InvalidOperationException("Invalid syntax");
	}

	private Operator ParseOperator()
	{
		int start = position;
		while (position < query.Length && char.IsLetter(query[position]))
		{
			position++;
		}

		var operatorName = query[start..position].ToString();
		return operatorName switch
		{
			"not" => Operator.Not,
			"and" => Operator.And,
			"or" => Operator.Or,
			_ => Operator.None,
		};
	}

	private string ParseTerm()
	{
		Expect('\'');
		int start = position;

		while (query[position] != '\'')
		{
			position++;
		}

		string term = query[start..position].ToString();
		Expect('\'');
		return term;
	}

	private void Expect(char expected)
	{
		SkipWhitespace();
		if (query[position] != expected)
			throw new InvalidOperationException($"Expected '{expected}' at position {position}");
		position++;
	}

	private void SkipWhitespace()
	{
		while (position < query.Length && char.IsWhiteSpace(query[position]))
		{
			position++;
		}
	}

	[GeneratedRegex(@"#q\d+=\s*(.*?);(?=\s*#q|\s*$)", RegexOptions.Singleline)]
	private static partial Regex QuerySplitter();

}
