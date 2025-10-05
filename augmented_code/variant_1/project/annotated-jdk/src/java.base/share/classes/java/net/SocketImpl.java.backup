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

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.FileDescriptor;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Objects;
import java.util.Set;
import sun.net.NetProperties;
import sun.net.PlatformSocketImpl;
import sun.nio.ch.NioSocketImpl;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class SocketImpl implements SocketOptions {

    @SuppressWarnings("unchecked")
    static <S extends SocketImpl & PlatformSocketImpl> S createPlatformSocketImpl(boolean server);

    protected FileDescriptor fd;

    protected InetAddress address;

    protected int port;

    protected int localport;

    public SocketImpl() {
    }

    protected abstract void create(boolean stream) throws IOException;

    protected abstract void connect(String host, int port) throws IOException;

    protected abstract void connect(InetAddress address, int port) throws IOException;

    protected abstract void connect(SocketAddress address, int timeout) throws IOException;

    protected abstract void bind(InetAddress host, int port) throws IOException;

    protected abstract void listen(int backlog) throws IOException;

    protected abstract void accept(SocketImpl s) throws IOException;

    protected abstract InputStream getInputStream() throws IOException;

    protected abstract OutputStream getOutputStream() throws IOException;

    protected abstract int available() throws IOException;

    protected abstract void close() throws IOException;

    void closeQuietly();

    protected void shutdownInput() throws IOException;

    protected void shutdownOutput() throws IOException;

    protected FileDescriptor getFileDescriptor();

    protected InetAddress getInetAddress();

    protected int getPort();

    protected boolean supportsUrgentData();

    protected abstract void sendUrgentData(int data) throws IOException;

    protected int getLocalPort();

    public String toString();

    void reset();

    protected void setPerformancePreferences(int connectionTime, int latency, int bandwidth);

    protected <T> void setOption(SocketOption<T> name, T value) throws IOException;

    protected <T> T getOption(SocketOption<T> name) throws IOException;

    void copyOptionsTo(SocketImpl target);

    protected Set<SocketOption<?>> supportedOptions();
}
