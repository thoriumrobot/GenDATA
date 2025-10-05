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
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import sun.security.util.SecurityConstants;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.OutputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.lang.invoke.MethodHandles;
    @Positive
import java.lang.invoke.VarHandle;
    @Positive
import java.nio.channels.SocketChannel;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import java.util.Collections;

    @Positive
@AnnotatedFor({ "calledmethods", "interning", "mustcall", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public class Socket implements java.io.Closeable {

    @Positive
    public Socket() {
    @Positive
    }

    @Positive
    public Socket(Proxy proxy) {
    @Positive
    }

    @Positive
    protected Socket(SocketImpl impl) throws SocketException {
    @Positive
    }

    @Positive
    public Socket(@Nullable String host, int port) throws UnknownHostException, IOException {
    @Positive
    }

    @Positive
    public Socket(InetAddress address, int port) throws IOException {
    @Positive
    }

    @Positive
    public Socket(@Nullable String host, int port, @Nullable InetAddress localAddr, int localPort) throws IOException {
    @Positive
    }

    @Positive
    public Socket(InetAddress address, int port, @Nullable InetAddress localAddr, int localPort) throws IOException {
    @Positive
    }

    @Positive
    @Deprecated
    @Positive
    public Socket(@Nullable String host, int port, boolean stream) throws IOException {
    @Positive
    }

    @Positive
    @Deprecated
    @Positive
    public Socket(InetAddress host, int port, boolean stream) throws IOException {
    @Positive
    }

    @Positive
    void createImpl(boolean stream) throws SocketException;

    @Positive
    void setImpl(SocketImpl si);

    @Positive
    void setImpl();

    @Positive
    SocketImpl getImpl() throws SocketException;

    @Positive
    @CreatesMustCallFor
    @Positive
    public void connect(SocketAddress endpoint) throws IOException;

    @Positive
    @CreatesMustCallFor
    @Positive
    public void connect(SocketAddress endpoint, int timeout) throws IOException;

    @Positive
    @CreatesMustCallFor
    @Positive
    public void bind(@Nullable SocketAddress bindpoint) throws IOException;

    @Positive
    final void postAccept();

    @Positive
    @Nullable
    @Positive
    public InetAddress getInetAddress();

    @Positive
    public InetAddress getLocalAddress();

    @Positive
    public int getPort();

    @Positive
    public int getLocalPort();

    @Positive
    @Nullable
    @Positive
    public SocketAddress getRemoteSocketAddress();

    @Positive
    @Nullable
    @Positive
    public SocketAddress getLocalSocketAddress();

    @Positive
    @Nullable
    @Positive
    @MustCallAlias
    @Positive
    public SocketChannel getChannel(@MustCallAlias Socket this);

    @Positive
    @MustCallAlias
    @Positive
    public InputStream getInputStream(@MustCallAlias Socket this) throws IOException;

    @Positive
    private static class SocketInputStream extends InputStream {

    @Positive
        @Override
    @Positive
        public int read() throws IOException;

    @Positive
        @Override
    @Positive
        public int read(byte[] b, int off, int len) throws IOException;

    @Positive
        @Override
    @Positive
        public int available() throws IOException;

    @Positive
        @Override
    @Positive
        public void close() throws IOException;
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public OutputStream getOutputStream(@MustCallAlias Socket this) throws IOException;

    @Positive
    private static class SocketOutputStream extends OutputStream {

    @Positive
        @Override
    @Positive
        public void write(int b) throws IOException;

    @Positive
        @Override
    @Positive
        public void write(byte[] b, int off, int len) throws IOException;

    @Positive
        @Override
    @Positive
        public void close() throws IOException;
    @Positive
    }

    @Positive
    public void setTcpNoDelay(boolean on) throws SocketException;

    @Positive
    public boolean getTcpNoDelay() throws SocketException;

    @Positive
    public void setSoLinger(boolean on, int linger) throws SocketException;

    @Positive
    public int getSoLinger() throws SocketException;

    @Positive
    public void sendUrgentData(int data) throws IOException;

    @Positive
    public void setOOBInline(boolean on) throws SocketException;

    @Positive
    public boolean getOOBInline() throws SocketException;

    @Positive
    public synchronized void setSoTimeout(int timeout) throws SocketException;

    @Positive
    public synchronized int getSoTimeout() throws SocketException;

    @Positive
    public synchronized void setSendBufferSize(int size) throws SocketException;

    @Positive
    public synchronized int getSendBufferSize() throws SocketException;

    @Positive
    public synchronized void setReceiveBufferSize(int size) throws SocketException;

    @Positive
    public synchronized int getReceiveBufferSize() throws SocketException;

    @Positive
    public void setKeepAlive(boolean on) throws SocketException;

    @Positive
    public boolean getKeepAlive() throws SocketException;

    @Positive
    public void setTrafficClass(int tc) throws SocketException;

    @Positive
    public int getTrafficClass() throws SocketException;

    @Positive
    public void setReuseAddress(boolean on) throws SocketException;

    @Positive
    public boolean getReuseAddress() throws SocketException;

    @Positive
    public synchronized void close() throws IOException;

    @Positive
    public void shutdownInput() throws IOException;

    @Positive
    public void shutdownOutput() throws IOException;

    @Positive
    public String toString();

    @Positive
    public boolean isConnected();

    @Positive
    public boolean isBound();

    @Positive
    @EnsuresCalledMethodsIf(expression = "this", result = true, methods = { "close" })
    @Positive
    public boolean isClosed();

    @Positive
    public boolean isInputShutdown();

    @Positive
    public boolean isOutputShutdown();

    @Positive
    static SocketImplFactory socketImplFactory();

    @Positive
    @Deprecated()
    @Positive
    public static synchronized void setSocketImplFactory(@Nullable SocketImplFactory fac) throws IOException;

    @Positive
    public void setPerformancePreferences(int connectionTime, int latency, int bandwidth);

    @Positive
    public <T> Socket setOption(SocketOption<T> name, T value) throws IOException;

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public <T> T getOption(SocketOption<T> name) throws IOException;

    @Positive
    public Set<SocketOption<?>> supportedOptions();
    @Positive
}

// CFWR semantic augmentation - variant 0
