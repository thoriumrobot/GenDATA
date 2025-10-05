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
package sun.nio.ch;

    @Positive
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.FileDescriptor;
    @Positive
import java.io.IOException;
    @Positive
import java.lang.reflect.Constructor;
    @Positive
import java.lang.reflect.InvocationTargetException;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.nio.MappedByteBuffer;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Set;
    @Positive
import jdk.internal.access.foreign.MemorySegmentProxy;
    @Positive
import jdk.internal.misc.TerminatingThreadLocal;
    @Positive
import jdk.internal.misc.Unsafe;
    @Positive
import sun.security.action.GetPropertyAction;

    @Positive
public class Util {

    @Positive
    private static class BufferCache {

    @Positive
        ByteBuffer get(int size);

    @Positive
        boolean offerFirst(ByteBuffer buf);

    @Positive
        boolean offerLast(ByteBuffer buf);

    @Positive
        boolean isEmpty();

    @Positive
        ByteBuffer removeFirst();
    @Positive
    }

    @Positive
    public static ByteBuffer getTemporaryDirectBuffer(int size);

    @Positive
    public static ByteBuffer getTemporaryAlignedDirectBuffer(int size, int alignment);

    @Positive
    public static void releaseTemporaryDirectBuffer(ByteBuffer buf);

    @Positive
    static void offerFirstTemporaryDirectBuffer(ByteBuffer buf);

    @Positive
    static void offerLastTemporaryDirectBuffer(ByteBuffer buf);

    @Positive
    static ByteBuffer[] subsequence(ByteBuffer[] bs, int offset, int length);

    @Positive
    static <E> Set<E> ungrowableSet(final Set<E> s);

    @Positive
    static void erase(ByteBuffer bb);

    @Positive
    static Unsafe unsafe();

    @Positive
    static int pageSize();

    @Positive
    static MappedByteBuffer newMappedByteBuffer(int size, long addr, FileDescriptor fd, Runnable unmapper, boolean isSync);

    @Positive
    static MappedByteBuffer newMappedByteBufferR(int size, long addr, FileDescriptor fd, Runnable unmapper, boolean isSync);

    @Positive
    static void checkBufferPositionAligned(ByteBuffer bb, int pos, int alignment) throws IOException;

    @Positive
    static void checkRemainingBufferSizeAligned(int rem, int alignment) throws IOException;

    @Positive
    static void checkChannelPositionAligned(long position, int alignment) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 0
