/*
 * Copyright (c) 1996, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.util.zip;

import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.lang.ref.Cleaner.Cleanable;
import java.lang.ref.Reference;
import java.nio.ByteBuffer;
import java.nio.ReadOnlyBufferException;
import java.util.Objects;
import jdk.internal.ref.CleanerFactory;
import sun.nio.ch.DirectBuffer;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class Deflater {

    public static final int DEFLATED;

    public static final int NO_COMPRESSION;

    public static final int BEST_SPEED;

    public static final int BEST_COMPRESSION;

    public static final int DEFAULT_COMPRESSION;

    public static final int FILTERED;

    public static final int HUFFMAN_ONLY;

    public static final int DEFAULT_STRATEGY;

    public static final int NO_FLUSH;

    public static final int SYNC_FLUSH;

    public static final int FULL_FLUSH;

    public Deflater(int level, boolean nowrap) {
    }

    public Deflater(int level) {
    }

    public Deflater() {
    }

    public void setInput(byte[] input, @IndexOrHigh({ "#1" }) int off, @IndexOrHigh({ "#1" }) int len);

    public void setInput(byte[] input);

    public void setInput(ByteBuffer input);

    public void setDictionary(byte[] dictionary, @IndexOrHigh({ "#1" }) int off, @IndexOrHigh({ "#1" }) int len);

    public void setDictionary(byte[] dictionary);

    public void setDictionary(ByteBuffer dictionary);

    public void setStrategy(int strategy);

    public void setLevel(int level);

    public boolean needsInput();

    public void finish();

    public boolean finished();

    @GTENegativeOne
    public int deflate(byte[] output, @IndexOrHigh({ "#1" }) int off, @IndexOrHigh({ "#1" }) int len);

    @GTENegativeOne
    public int deflate(byte[] output);

    public int deflate(ByteBuffer output);

    @GTENegativeOne
    public int deflate(byte[] output, @IndexOrHigh({ "#1" }) int off, @IndexOrHigh({ "#1" }) int len, int flush);

    public int deflate(ByteBuffer output, int flush);

    public int getAdler();

    public int getTotalIn();

    public long getBytesRead();

    public int getTotalOut();

    public long getBytesWritten();

    public void reset();

    public void end();

    static class DeflaterZStreamRef implements Runnable {

        long address();

        void clean();

        public synchronized void run();
    }
}
