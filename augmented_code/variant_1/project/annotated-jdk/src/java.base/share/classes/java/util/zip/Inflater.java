/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2018, Oracle and/or its affiliates. All rights reserved.
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
package java.util.zip;

    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.ref.Cleaner.Cleanable;
    @Positive
import java.lang.ref.Reference;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.nio.ReadOnlyBufferException;
    @Positive
import java.util.Objects;
    @Positive
import jdk.internal.ref.CleanerFactory;
    @Positive
import sun.nio.ch.DirectBuffer;

    @Positive
@AnnotatedFor({ "index", "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class Inflater {

    @Positive
    public Inflater(boolean nowrap) {
    @Positive
    }

    @Positive
    public Inflater() {
    @Positive
    }

    @Positive
    public void setInput(byte[] input, @IndexOrHigh({ "#1" }) int off, @IndexOrHigh({ "#1" }) int len);

    @Positive
    public void setInput(byte[] input);

    @Positive
    public void setInput(ByteBuffer input);

    @Positive
    public void setDictionary(byte[] dictionary, @IndexOrHigh({ "#1" }) int off, @IndexOrHigh({ "#1" }) int len);

    @Positive
    public void setDictionary(byte[] dictionary);

    @Positive
    public void setDictionary(ByteBuffer dictionary);

    @Positive
    @NonNegative
    @Positive
    public int getRemaining();

    @Positive
    public boolean needsInput();

    @Positive
    public boolean needsDictionary();

    @Positive
    public boolean finished();

    @Positive
    @GTENegativeOne
    @Positive
    public int inflate(byte[] output, @IndexOrHigh({ "#1" }) int off, @IndexOrHigh({ "#1" }) int len) throws DataFormatException;

    @Positive
    @GTENegativeOne
    @Positive
    public int inflate(byte[] output) throws DataFormatException;

    @Positive
    public int inflate(ByteBuffer output) throws DataFormatException;

    @Positive
    public int getAdler();

    @Positive
    public int getTotalIn();

    @Positive
    public long getBytesRead();

    @Positive
    public int getTotalOut();

    @Positive
    public long getBytesWritten();

    @Positive
    public void reset();

    @Positive
    public void end();

    @Positive
    static class InflaterZStreamRef implements Runnable {

    @Positive
        long address();

    @Positive
        void clean();

    @Positive
        public synchronized void run();
    @Positive
    }
    @Positive
}
