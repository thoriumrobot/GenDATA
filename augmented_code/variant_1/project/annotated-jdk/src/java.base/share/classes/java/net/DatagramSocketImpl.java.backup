/*
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.FileDescriptor;
import java.io.IOException;
import java.util.Objects;
import java.util.Set;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class DatagramSocketImpl implements SocketOptions {

    public DatagramSocketImpl() {
    }

    protected int localPort;

    protected FileDescriptor fd;

    int dataAvailable();

    protected abstract void create() throws SocketException;

    protected abstract void bind(int lport, InetAddress laddr) throws SocketException;

    protected abstract void send(DatagramPacket p) throws IOException;

    protected void connect(InetAddress address, int port) throws SocketException;

    protected void disconnect();

    @Pure
    protected abstract int peek(InetAddress i) throws IOException;

    protected abstract int peekData(DatagramPacket p) throws IOException;

    protected abstract void receive(DatagramPacket p) throws IOException;

    @Deprecated
    protected abstract void setTTL(byte ttl) throws IOException;

    @Deprecated
    protected abstract byte getTTL() throws IOException;

    protected abstract void setTimeToLive(int ttl) throws IOException;

    protected abstract int getTimeToLive() throws IOException;

    protected abstract void join(InetAddress inetaddr) throws IOException;

    protected abstract void leave(InetAddress inetaddr) throws IOException;

    protected abstract void joinGroup(SocketAddress mcastaddr, NetworkInterface netIf) throws IOException;

    protected abstract void leaveGroup(SocketAddress mcastaddr, NetworkInterface netIf) throws IOException;

    protected abstract void close();

    protected int getLocalPort();

    protected FileDescriptor getFileDescriptor();

    protected <T> void setOption(SocketOption<T> name, T value) throws IOException;

    protected <T> T getOption(SocketOption<T> name) throws IOException;

    protected Set<SocketOption<?>> supportedOptions();
}
