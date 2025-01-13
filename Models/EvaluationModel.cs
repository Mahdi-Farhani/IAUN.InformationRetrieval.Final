namespace IAUN.InformationRetrieval.Final.Models;

public class EvaluationModel
{
	
	public int QueryIndex { get; set; }
	public string Result { get; set; } = string.Empty;
	public decimal Precision { get; set; }
	public decimal ReCall { get; set; }
	public decimal FMeasure { get; set; }
}
