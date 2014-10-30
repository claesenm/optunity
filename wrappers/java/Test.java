// this is a simple test case for communicating with Optunity over sockets in Java

import java.io.*;
import java.net.*;
import java.util.Arrays;

public class Test {
    public static void main(String args[]) throws Exception {
        String pythonPath = System.getenv("PYTHONPATH");
        System.out.println("Current PYTHONPATH: " + pythonPath);

        String msg = "{\"manual\": \"\"}";

        ServerSocket servSock = new ServerSocket(0);
        System.out.println("Server socket bound to port: " + servSock.getLocalPort());

        String[] cmd = new String[] {"python","-m","optunity.piped",""
            + servSock.getLocalPort()};
        String[] pyEnv = new String[1];

        final String dir = System.getProperty("user.dir");
        pyEnv[0] = "PYTHONPATH=" + pythonPath + System.getProperty("path.separator")
            + dir.substring(0, dir.length()-13); // length("wrappers/java") == 13
        System.out.println("Attempting to launch: " + Arrays.toString(cmd)
                + " with env " + Arrays.toString(pyEnv));

        Process p = Runtime.getRuntime().exec(cmd, pyEnv);
        System.out.println("Successfully launched Python subprocess. Connecting ...");

        Socket connSock = servSock.accept();
        System.out.println("Connection established.");

        BufferedReader py2java = new BufferedReader(new InputStreamReader(connSock.getInputStream()));
        PrintWriter java2py = new PrintWriter(connSock.getOutputStream());

        System.out.println("Sending message to Python: " + msg);
        java2py.println(msg);
        java2py.flush();

        System.out.println("Waiting for reply.");
        String reply = py2java.readLine();
        System.out.println("Received: " + reply);
    }
}
