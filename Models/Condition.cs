namespace IAUN.InformationRetrieval.Final.Models;

public abstract class QueryNode
{

}
public class LogicalNode : QueryNode
{
	public Operator Operator { get; set; } = Operator.None;
	public List<QueryNode> Operands { get; set; } = [];
}
public class TermNode:QueryNode
{
	public string Term { get; set; } = string.Empty;
}

public enum Operator
{
	None = 0, Not = 1, And = 2, Or = 3
}