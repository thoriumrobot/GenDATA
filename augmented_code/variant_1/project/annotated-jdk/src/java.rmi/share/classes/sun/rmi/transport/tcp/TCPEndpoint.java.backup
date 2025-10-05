/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package sun.rmi.transport.tcp;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.lang.reflect.Proxy;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.rmi.ConnectIOException;
import java.rmi.RemoteException;
import java.rmi.server.RMIClientSocketFactory;
import java.rmi.server.RMIServerSocketFactory;
import java.rmi.server.RMISocketFactory;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;
import sun.rmi.runtime.Log;
import sun.rmi.runtime.NewThreadAction;
import sun.rmi.transport.Channel;
import sun.rmi.transport.Endpoint;
import sun.rmi.transport.Target;
import sun.rmi.transport.Transport;

public class TCPEndpoint implements Endpoint {

    public TCPEndpoint(String host, int port) {
    }

    public TCPEndpoint(String host, int port, RMIClientSocketFactory csf, RMIServerSocketFactory ssf) {
    }

    public static TCPEndpoint getLocalEndpoint(int port);

    public static TCPEndpoint getLocalEndpoint(int port, RMIClientSocketFactory csf, RMIServerSocketFactory ssf);

    static void setLocalHost(String host);

    static void setDefaultPort(int port, RMIClientSocketFactory csf, RMIServerSocketFactory ssf);

    public Transport getOutboundTransport();

    public static void shedConnectionCaches();

    public void exportObject(Target target) throws RemoteException;

    public Channel getChannel();

    public String getHost();

    public int getPort();

    public int getListenPort();

    public Transport getInboundTransport();

    public RMIClientSocketFactory getClientSocketFactory();

    public RMIServerSocketFactory getServerSocketFactory();

    public String toString();

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public void write(ObjectOutput out) throws IOException;

    public static TCPEndpoint read(ObjectInput in) throws IOException, ClassNotFoundException;

    public void writeHostPortFormat(DataOutput out) throws IOException;

    public static TCPEndpoint readHostPortFormat(DataInput in) throws IOException;

    Socket newSocket() throws RemoteException;

    ServerSocket newServerSocket() throws IOException;

    private static class FQDN implements Runnable {

        static String attemptFQDN(InetAddress localAddr) throws java.net.UnknownHostException;

        public void run();
    }
}
