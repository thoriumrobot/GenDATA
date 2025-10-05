/*
    @Positive
 * Copyright (c) 2012, 2013, Oracle and/or its affiliates. All rights reserved.
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
package java.util.stream;

    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.dataflow.qual.Pure;

    @Positive
abstract class AbstractSpinedBuffer {

    @Positive
    public static final int MIN_CHUNK_POWER;

    @Positive
    public static final int MIN_CHUNK_SIZE;

    @Positive
    public static final int MAX_CHUNK_POWER;

    @Positive
    public static final int MIN_SPINE_SIZE;

    @Positive
    protected final int initialChunkPower;

    @Positive
    protected int elementIndex;

    @Positive
    protected int spineIndex;

    @Positive
    protected long[] priorElementCount;

    @Positive
    protected AbstractSpinedBuffer() {
    @Positive
    }

    @Positive
    protected AbstractSpinedBuffer(int initialCapacity) {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    public boolean isEmpty();

    @Positive
    public long count();

    @Positive
    protected int chunkSize(int n);

    @Positive
    public abstract void clear();
    @Positive
}

// CFWR semantic augmentation - variant 0
