/*
 * Copyright (c) 2005, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.io;

import org.checkerframework.checker.formatter.qual.FormatMethod;
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.*;
import java.nio.charset.Charset;
import jdk.internal.access.JavaIOAccess;
import jdk.internal.access.SharedSecrets;
import sun.nio.cs.StreamDecoder;
import sun.nio.cs.StreamEncoder;
import sun.security.action.GetPropertyAction;

@AnnotatedFor({ "formatter", "index", "interning", "nullness" })
@UsesObjectEquals
public final class Console implements Flushable {

    public PrintWriter writer();

    public Reader reader();

    @FormatMethod
    public Console format(String fmt, @Nullable Object... args);

    @FormatMethod
    public Console printf(String format, @Nullable Object... args);

    @Nullable
    public String readLine(String fmt, @Nullable Object... args);

    @Nullable
    public String readLine();

    public char @Nullable [] readPassword(String fmt, @Nullable Object... args);

    public char @Nullable [] readPassword();

    public void flush();

    @Pure
    public Charset charset();

    class LineReader extends Reader {

        public void close();

        public boolean ready() throws IOException;

        @GTENegativeOne
        @LTEqLengthOf({ "#1" })
        public int read(char[] cbuf, @IndexOrHigh({ "#1" }) int offset, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int length) throws IOException;
    }
}
