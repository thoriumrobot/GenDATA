/*
    @Positive
 * Copyright (c) 2005, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.formatter.qual.FormatMethod;
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.*;
    @Positive
import java.nio.charset.Charset;
    @Positive
import jdk.internal.access.JavaIOAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import sun.nio.cs.StreamDecoder;
    @Positive
import sun.nio.cs.StreamEncoder;
    @Positive
import sun.security.action.GetPropertyAction;

    @Positive
@AnnotatedFor({ "formatter", "index", "interning", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public final class Console implements Flushable {

    @Positive
    public PrintWriter writer();

    @Positive
    public Reader reader();

    @Positive
    @FormatMethod
    @Positive
    public Console format(String fmt, @Nullable Object... args);

    @Positive
    @FormatMethod
    @Positive
    public Console printf(String format, @Nullable Object... args);

    @Positive
    @Nullable
    @Positive
    public String readLine(String fmt, @Nullable Object... args);

    @Positive
    @Nullable
    @Positive
    public String readLine();

    @Positive
    public char @Nullable [] readPassword(String fmt, @Nullable Object... args);

    @Positive
    public char @Nullable [] readPassword();

    @Positive
    public void flush();

    @Positive
    @Pure
    @Positive
    public Charset charset();

    @Positive
    class LineReader extends Reader {

    @Positive
        public void close();

    @Positive
        public boolean ready() throws IOException;

    @Positive
        @GTENegativeOne
    @Positive
        @LTEqLengthOf({ "#1" })
    @Positive
        public int read(char[] cbuf, @IndexOrHigh({ "#1" }) int offset, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int length) throws IOException;
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
