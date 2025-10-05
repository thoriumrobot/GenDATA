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
package java.nio;

    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.LessThan;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import jdk.internal.access.JavaNioAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.access.foreign.MemorySegmentProxy;
    @Positive
import jdk.internal.access.foreign.UnmapperProxy;
    @Positive
import jdk.internal.misc.ScopedMemoryAccess;
    @Positive
import jdk.internal.misc.ScopedMemoryAccess.Scope;
    @Positive
import jdk.internal.misc.Unsafe;
    @Positive
import jdk.internal.misc.VM.BufferPool;
    @Positive
import jdk.internal.vm.annotation.ForceInline;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;
    @Positive
import java.io.FileDescriptor;
    @Positive
import java.util.Spliterator;

    @Positive
@AnnotatedFor({ "index" })
    @Positive
public abstract class Buffer {

    @Positive
    static IllegalArgumentException createSameBufferException();

    @Positive
    static IllegalArgumentException createCapacityException(int capacity);

    @Positive
    @NonNegative
    @Positive
    public final int capacity();

    @Positive
    @NonNegative
    @Positive
    public final int position();

    @Positive
    public Buffer position(@NonNegative int newPosition);

    @Positive
    @NonNegative
    @Positive
    public final int limit();

    @Positive
    public Buffer limit(@NonNegative int newLimit);

    @Positive
    public Buffer mark();

    @Positive
    public Buffer reset();

    @Positive
    public Buffer clear();

    @Positive
    public Buffer flip();

    @Positive
    public Buffer rewind();

    @Positive
    @NonNegative
    @Positive
    public final int remaining();

    @Positive
    public final boolean hasRemaining();

    @Positive
    public abstract boolean isReadOnly();

    @Positive
    public abstract boolean hasArray();

    @Positive
    public abstract Object array();

    @Positive
    @NonNegative
    @Positive
    public abstract int arrayOffset();

    @Positive
    public abstract boolean isDirect();

    @Positive
    public abstract Buffer slice();

    @Positive
    public abstract Buffer slice(int index, int length);

    @Positive
    public abstract Buffer duplicate();

    @Positive
    abstract Object base();

    @Positive
    final int nextGetIndex();

    @Positive
    final int nextGetIndex(int nb);

    @Positive
    final int nextPutIndex();

    @Positive
    final int nextPutIndex(int nb);

    @Positive
    @IntrinsicCandidate
    @Positive
    final int checkIndex(int i);

    @Positive
    final int checkIndex(int i, int nb);

    @Positive
    @GTENegativeOne
    @Positive
    final int markValue();

    @Positive
    final void discardMark();

    @Positive
    @ForceInline
    @Positive
    final ScopedMemoryAccess.Scope scope();

    @Positive
    final void checkScope();
    @Positive
}

// CFWR semantic augmentation - variant 0
