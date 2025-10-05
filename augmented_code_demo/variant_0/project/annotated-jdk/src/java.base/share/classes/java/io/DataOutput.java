/*
    @Positive
 * Copyright (c) 1995, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.io;

    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;

    @Positive
@AnnotatedFor({ "nullness", "index" })
    @Positive
public interface DataOutput {

    @Positive
    void write(int b) throws IOException;

    @Positive
    void write(byte[] b) throws IOException;

    @Positive
    void write(@PolySigned byte[] b, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len) throws IOException;

    @Positive
    void writeBoolean(boolean v) throws IOException;

    @Positive
    void writeByte(int v) throws IOException;

    @Positive
    void writeShort(int v) throws IOException;

    @Positive
    void writeChar(int v) throws IOException;

    @Positive
    void writeInt(int v) throws IOException;

    @Positive
    void writeLong(long v) throws IOException;

    @Positive
    void writeFloat(float v) throws IOException;

    @Positive
    void writeDouble(double v) throws IOException;

    @Positive
    void writeBytes(String s) throws IOException;

    @Positive
    void writeChars(String s) throws IOException;

    @Positive
    void writeUTF(String s) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 0
