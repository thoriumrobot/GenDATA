/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.nio.channels.spi;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.lang.reflect.InvocationTargetException;
    @Positive
import java.net.ProtocolFamily;
    @Positive
import java.nio.channels.Channel;
    @Positive
import java.nio.channels.DatagramChannel;
    @Positive
import java.nio.channels.Pipe;
    @Positive
import java.nio.channels.ServerSocketChannel;
    @Positive
import java.nio.channels.SocketChannel;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Objects;
    @Positive
import java.util.ServiceLoader;
    @Positive
import java.util.ServiceConfigurationError;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class SelectorProvider {

    @Positive
    protected SelectorProvider() {
    @Positive
    }

    @Positive
    private static class Holder {

    @Positive
        @SuppressWarnings("removal")
    @Positive
        static SelectorProvider provider();
    @Positive
    }

    @Positive
    public static SelectorProvider provider();

    @Positive
    public abstract DatagramChannel openDatagramChannel() throws IOException;

    @Positive
    public abstract DatagramChannel openDatagramChannel(ProtocolFamily family) throws IOException;

    @Positive
    public abstract Pipe openPipe() throws IOException;

    @Positive
    public abstract AbstractSelector openSelector() throws IOException;

    @Positive
    public abstract ServerSocketChannel openServerSocketChannel() throws IOException;

    @Positive
    public abstract SocketChannel openSocketChannel() throws IOException;

    @Positive
    public Channel inheritedChannel() throws IOException;

    @Positive
    public SocketChannel openSocketChannel(ProtocolFamily family) throws IOException;

    @Positive
    public ServerSocketChannel openServerSocketChannel(ProtocolFamily family) throws IOException;
    @Positive
}
