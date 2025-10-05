/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.FileDescriptor;
    @Positive
import java.io.IOException;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class DatagramSocketImpl implements SocketOptions {

    @Positive
    public DatagramSocketImpl() {
    @Positive
    }

    @Positive
    protected int localPort;

    @Positive
    protected FileDescriptor fd;

    @Positive
    int dataAvailable();

    @Positive
    protected abstract void create() throws SocketException;

    @Positive
    protected abstract void bind(int lport, InetAddress laddr) throws SocketException;

    @Positive
    protected abstract void send(DatagramPacket p) throws IOException;

    @Positive
    protected void connect(InetAddress address, int port) throws SocketException;

    @Positive
    protected void disconnect();

    @Positive
    @Pure
    @Positive
    protected abstract int peek(InetAddress i) throws IOException;

    @Positive
    protected abstract int peekData(DatagramPacket p) throws IOException;

    @Positive
    protected abstract void receive(DatagramPacket p) throws IOException;

    @Positive
    @Deprecated
    @Positive
    protected abstract void setTTL(byte ttl) throws IOException;

    @Positive
    @Deprecated
    @Positive
    protected abstract byte getTTL() throws IOException;

    @Positive
    protected abstract void setTimeToLive(int ttl) throws IOException;

    @Positive
    protected abstract int getTimeToLive() throws IOException;

    @Positive
    protected abstract void join(InetAddress inetaddr) throws IOException;

    @Positive
    protected abstract void leave(InetAddress inetaddr) throws IOException;

    @Positive
    protected abstract void joinGroup(SocketAddress mcastaddr, NetworkInterface netIf) throws IOException;

    @Positive
    protected abstract void leaveGroup(SocketAddress mcastaddr, NetworkInterface netIf) throws IOException;

    @Positive
    protected abstract void close();

    @Positive
    protected int getLocalPort();

    @Positive
    protected FileDescriptor getFileDescriptor();

    @Positive
    protected <T> void setOption(SocketOption<T> name, T value) throws IOException;

    @Positive
    protected <T> T getOption(SocketOption<T> name) throws IOException;

    @Positive
    protected Set<SocketOption<?>> supportedOptions();
    @Positive
}
