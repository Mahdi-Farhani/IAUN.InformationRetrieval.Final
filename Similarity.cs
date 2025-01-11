namespace IAUN.InformationRetrieval.Final;

public class Similarity
{
	public static float CosineSimilarity(float[] vectorA, float[] vectorB)
	{
		float dotProduct = 0;
		float magnitudeA = 0;
		float magnitudeB = 0;

		for (int i = 0; i < vectorA.Length; i++)
		{
			dotProduct += vectorA[i] * vectorB[i];
			magnitudeA += vectorA[i] * vectorA[i];
			magnitudeB += vectorB[i] * vectorB[i];
		}

		return dotProduct / (float)(Math.Sqrt(magnitudeA) * Math.Sqrt(magnitudeB));
	}
}
