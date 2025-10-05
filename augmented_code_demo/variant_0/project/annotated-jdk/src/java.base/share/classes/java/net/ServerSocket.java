/*
    @Positive
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.net;

    @Positive
import org.checkerframework.checker.calledmethods.qual.EnsuresCalledMethodsIf;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.mustcall.qual.CreatesMustCallFor;
    @Positive
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.FileDescriptor;
    @Positive
import java.io.IOException;
    @Positive
import java.nio.channels.ServerSocketChannel;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import java.util.Collections;
    @Positive
import sun.security.util.SecurityConstants;
    @Positive
import sun.net.PlatformSocketImpl;

    @Positive
@AnnotatedFor({ "calledmethods", "interning", "mustcall" })
    @Positive
@UsesObjectEquals
    @Positive
public class ServerSocket implements java.io.Closeable {

    @Positive
    protected ServerSocket(SocketImpl impl) {
    @Positive
    }

    @Positive
    public ServerSocket() throws IOException {
    @Positive
    }

    @Positive
    public ServerSocket(int port) throws IOException {
    @Positive
    }

    @Positive
    public ServerSocket(int port, int backlog) throws IOException {
    @Positive
    }

    @Positive
    public ServerSocket(int port, int backlog, InetAddress bindAddr) throws IOException {
    @Positive
    }

    @Positive
    SocketImpl getImpl() throws SocketException;

    @Positive
    void createImpl() throws SocketException;

    @Positive
    @CreatesMustCallFor
    @Positive
    public void bind(SocketAddress endpoint) throws IOException;

    @Positive
    @CreatesMustCallFor
    @Positive
    public void bind(SocketAddress endpoint, int backlog) throws IOException;

    @Positive
    public InetAddress getInetAddress();

    @Positive
    public int getLocalPort();

    @Positive
    public SocketAddress getLocalSocketAddress();

    @Positive
    public Socket accept() throws IOException;

    @Positive
    protected final void implAccept(Socket s) throws IOException;

    @Positive
    public void close() throws IOException;

    @Positive
    @MustCallAlias
    @Positive
    public ServerSocketChannel getChannel(@MustCallAlias ServerSocket this);

    @Positive
    public boolean isBound();

    @Positive
    @EnsuresCalledMethodsIf(expression = "this", result = true, methods = { "close" })
    @Positive
    public boolean isClosed();

    @Positive
    public synchronized void setSoTimeout(int timeout) throws SocketException;

    @Positive
    public synchronized int getSoTimeout() throws IOException;

    @Positive
    public void setReuseAddress(boolean on) throws SocketException;

    @Positive
    public boolean getReuseAddress() throws SocketException;

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public String toString();

    @Positive
    @Deprecated()
    @Positive
    public static synchronized void setSocketFactory(SocketImplFactory fac) throws IOException;

    @Positive
    public synchronized void setReceiveBufferSize(int size) throws SocketException;

    @Positive
    public synchronized int getReceiveBufferSize() throws SocketException;

    @Positive
    public void setPerformancePreferences(int connectionTime, int latency, int bandwidth);

    @Positive
    public <T> ServerSocket setOption(SocketOption<T> name, T value) throws IOException;

    @Positive
    public <T> T getOption(SocketOption<T> name) throws IOException;

    @Positive
    public Set<SocketOption<?>> supportedOptions();
    @Positive
}

// CFWR semantic augmentation - variant 0
