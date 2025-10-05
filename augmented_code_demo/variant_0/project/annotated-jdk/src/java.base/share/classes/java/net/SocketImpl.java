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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.FileDescriptor;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.OutputStream;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import sun.net.NetProperties;
    @Positive
import sun.net.PlatformSocketImpl;
    @Positive
import sun.nio.ch.NioSocketImpl;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class SocketImpl implements SocketOptions {

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    static <S extends SocketImpl & PlatformSocketImpl> S createPlatformSocketImpl(boolean server);

    @Positive
    protected FileDescriptor fd;

    @Positive
    protected InetAddress address;

    @Positive
    protected int port;

    @Positive
    protected int localport;

    @Positive
    public SocketImpl() {
    @Positive
    }

    @Positive
    protected abstract void create(boolean stream) throws IOException;

    @Positive
    protected abstract void connect(String host, int port) throws IOException;

    @Positive
    protected abstract void connect(InetAddress address, int port) throws IOException;

    @Positive
    protected abstract void connect(SocketAddress address, int timeout) throws IOException;

    @Positive
    protected abstract void bind(InetAddress host, int port) throws IOException;

    @Positive
    protected abstract void listen(int backlog) throws IOException;

    @Positive
    protected abstract void accept(SocketImpl s) throws IOException;

    @Positive
    protected abstract InputStream getInputStream() throws IOException;

    @Positive
    protected abstract OutputStream getOutputStream() throws IOException;

    @Positive
    protected abstract int available() throws IOException;

    @Positive
    protected abstract void close() throws IOException;

    @Positive
    void closeQuietly();

    @Positive
    protected void shutdownInput() throws IOException;

    @Positive
    protected void shutdownOutput() throws IOException;

    @Positive
    protected FileDescriptor getFileDescriptor();

    @Positive
    protected InetAddress getInetAddress();

    @Positive
    protected int getPort();

    @Positive
    protected boolean supportsUrgentData();

    @Positive
    protected abstract void sendUrgentData(int data) throws IOException;

    @Positive
    protected int getLocalPort();

    @Positive
    public String toString();

    @Positive
    void reset();

    @Positive
    protected void setPerformancePreferences(int connectionTime, int latency, int bandwidth);

    @Positive
    protected <T> void setOption(SocketOption<T> name, T value) throws IOException;

    @Positive
    protected <T> T getOption(SocketOption<T> name) throws IOException;

    @Positive
    void copyOptionsTo(SocketImpl target);

    @Positive
    protected Set<SocketOption<?>> supportedOptions();
    @Positive
}

// CFWR semantic augmentation - variant 0
