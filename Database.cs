using StackExchange.Redis;
using System.Net;

namespace IAUN.InformationRetrieval.Final;

public class Database
{
	private static Lazy<ConnectionMultiplexer> connection = CreateConnection();

	public static ConnectionMultiplexer Connection
	{
		get { return connection.Value; }
	}
	private static Lazy<ConnectionMultiplexer> CreateConnection()
	{
		return new Lazy<ConnectionMultiplexer>(() =>
		{
			return ConnectionMultiplexer.Connect("localhost:6381");
		});
	}
	public static IDatabase GetDatabase()
	{
		return Connection.GetDatabase();
	}
	public static EndPoint[] GetEndPoints()
	{
		return Connection.GetEndPoints();
	}
	public static IServer GetServer(string host, int port)
	{
		return Connection.GetServer(host, port);
	}

}
