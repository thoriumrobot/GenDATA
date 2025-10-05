/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package sun.rmi.transport.tcp;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.io.DataInput;
    @Positive
import java.io.DataOutput;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInput;
    @Positive
import java.io.ObjectOutput;
    @Positive
import java.lang.reflect.Proxy;
    @Positive
import java.net.InetAddress;
    @Positive
import java.net.ServerSocket;
    @Positive
import java.net.Socket;
    @Positive
import java.rmi.ConnectIOException;
    @Positive
import java.rmi.RemoteException;
    @Positive
import java.rmi.server.RMIClientSocketFactory;
    @Positive
import java.rmi.server.RMIServerSocketFactory;
    @Positive
import java.rmi.server.RMISocketFactory;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.Collection;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.LinkedList;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;
    @Positive
import sun.rmi.runtime.Log;
    @Positive
import sun.rmi.runtime.NewThreadAction;
    @Positive
import sun.rmi.transport.Channel;
    @Positive
import sun.rmi.transport.Endpoint;
    @Positive
import sun.rmi.transport.Target;
    @Positive
import sun.rmi.transport.Transport;

    @Positive
public class TCPEndpoint implements Endpoint {

    @Positive
    public TCPEndpoint(String host, int port) {
    @Positive
    }

    @Positive
    public TCPEndpoint(String host, int port, RMIClientSocketFactory csf, RMIServerSocketFactory ssf) {
    @Positive
    }

    @Positive
    public static TCPEndpoint getLocalEndpoint(int port);

    @Positive
    public static TCPEndpoint getLocalEndpoint(int port, RMIClientSocketFactory csf, RMIServerSocketFactory ssf);

    @Positive
    static void setLocalHost(String host);

    @Positive
    static void setDefaultPort(int port, RMIClientSocketFactory csf, RMIServerSocketFactory ssf);

    @Positive
    public Transport getOutboundTransport();

    @Positive
    public static void shedConnectionCaches();

    @Positive
    public void exportObject(Target target) throws RemoteException;

    @Positive
    public Channel getChannel();

    @Positive
    public String getHost();

    @Positive
    public int getPort();

    @Positive
    public int getListenPort();

    @Positive
    public Transport getInboundTransport();

    @Positive
    public RMIClientSocketFactory getClientSocketFactory();

    @Positive
    public RMIServerSocketFactory getServerSocketFactory();

    @Positive
    public String toString();

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public void write(ObjectOutput out) throws IOException;

    @Positive
    public static TCPEndpoint read(ObjectInput in) throws IOException, ClassNotFoundException;

    @Positive
    public void writeHostPortFormat(DataOutput out) throws IOException;

    @Positive
    public static TCPEndpoint readHostPortFormat(DataInput in) throws IOException;

    @Positive
    Socket newSocket() throws RemoteException;

    @Positive
    ServerSocket newServerSocket() throws IOException;

    @Positive
    private static class FQDN implements Runnable {

    @Positive
        static String attemptFQDN(InetAddress localAddr) throws java.net.UnknownHostException;

    @Positive
        public void run();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
