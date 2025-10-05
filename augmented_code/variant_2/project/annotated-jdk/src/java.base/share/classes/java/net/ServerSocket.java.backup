/*
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.net;

import org.checkerframework.checker.calledmethods.qual.EnsuresCalledMethodsIf;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.mustcall.qual.CreatesMustCallFor;
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.FileDescriptor;
import java.io.IOException;
import java.nio.channels.ServerSocketChannel;
import java.util.Objects;
import java.util.Set;
import java.util.Collections;
import sun.security.util.SecurityConstants;
import sun.net.PlatformSocketImpl;

@AnnotatedFor({ "calledmethods", "interning", "mustcall" })
@UsesObjectEquals
public class ServerSocket implements java.io.Closeable {

    protected ServerSocket(SocketImpl impl) {
    }

    public ServerSocket() throws IOException {
    }

    public ServerSocket(int port) throws IOException {
    }

    public ServerSocket(int port, int backlog) throws IOException {
    }

    public ServerSocket(int port, int backlog, InetAddress bindAddr) throws IOException {
    }

    SocketImpl getImpl() throws SocketException;

    void createImpl() throws SocketException;

    @CreatesMustCallFor
    public void bind(SocketAddress endpoint) throws IOException;

    @CreatesMustCallFor
    public void bind(SocketAddress endpoint, int backlog) throws IOException;

    public InetAddress getInetAddress();

    public int getLocalPort();

    public SocketAddress getLocalSocketAddress();

    public Socket accept() throws IOException;

    protected final void implAccept(Socket s) throws IOException;

    public void close() throws IOException;

    @MustCallAlias
    public ServerSocketChannel getChannel(@MustCallAlias ServerSocket this);

    public boolean isBound();

    @EnsuresCalledMethodsIf(expression = "this", result = true, methods = { "close" })
    public boolean isClosed();

    public synchronized void setSoTimeout(int timeout) throws SocketException;

    public synchronized int getSoTimeout() throws IOException;

    public void setReuseAddress(boolean on) throws SocketException;

    public boolean getReuseAddress() throws SocketException;

    @SuppressWarnings("removal")
    public String toString();

    @Deprecated()
    public static synchronized void setSocketFactory(SocketImplFactory fac) throws IOException;

    public synchronized void setReceiveBufferSize(int size) throws SocketException;

    public synchronized int getReceiveBufferSize() throws SocketException;

    public void setPerformancePreferences(int connectionTime, int latency, int bandwidth);

    public <T> ServerSocket setOption(SocketOption<T> name, T value) throws IOException;

    public <T> T getOption(SocketOption<T> name) throws IOException;

    public Set<SocketOption<?>> supportedOptions();
}
