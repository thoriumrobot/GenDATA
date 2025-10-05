/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.nio.ch;

import org.checkerframework.checker.signedness.qual.UnknownSignedness;
import org.checkerframework.dataflow.qual.Pure;
import java.io.FileDescriptor;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Collection;
import java.util.Iterator;
import java.util.Set;
import jdk.internal.access.foreign.MemorySegmentProxy;
import jdk.internal.misc.TerminatingThreadLocal;
import jdk.internal.misc.Unsafe;
import sun.security.action.GetPropertyAction;

public class Util {

    private static class BufferCache {

        ByteBuffer get(int size);

        boolean offerFirst(ByteBuffer buf);

        boolean offerLast(ByteBuffer buf);

        boolean isEmpty();

        ByteBuffer removeFirst();
    }

    public static ByteBuffer getTemporaryDirectBuffer(int size);

    public static ByteBuffer getTemporaryAlignedDirectBuffer(int size, int alignment);

    public static void releaseTemporaryDirectBuffer(ByteBuffer buf);

    static void offerFirstTemporaryDirectBuffer(ByteBuffer buf);

    static void offerLastTemporaryDirectBuffer(ByteBuffer buf);

    static ByteBuffer[] subsequence(ByteBuffer[] bs, int offset, int length);

    static <E> Set<E> ungrowableSet(final Set<E> s);

    static void erase(ByteBuffer bb);

    static Unsafe unsafe();

    static int pageSize();

    static MappedByteBuffer newMappedByteBuffer(int size, long addr, FileDescriptor fd, Runnable unmapper, boolean isSync);

    static MappedByteBuffer newMappedByteBufferR(int size, long addr, FileDescriptor fd, Runnable unmapper, boolean isSync);

    static void checkBufferPositionAligned(ByteBuffer bb, int pos, int alignment) throws IOException;

    static void checkRemainingBufferSizeAligned(int rem, int alignment) throws IOException;

    static void checkChannelPositionAligned(long position, int alignment) throws IOException;
}
