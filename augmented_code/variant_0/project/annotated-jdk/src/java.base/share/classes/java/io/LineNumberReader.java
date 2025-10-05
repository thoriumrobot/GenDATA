/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;

    @Positive
@AnnotatedFor({ "index", "lock", "mustcall", "nullness" })
    @Positive
public class LineNumberReader extends BufferedReader {

    @Positive
    @MustCallAlias
    @Positive
    public LineNumberReader(@MustCallAlias Reader in) {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public LineNumberReader(@MustCallAlias Reader in, @NonNegative int sz) {
    @Positive
    }

    @Positive
    public void setLineNumber(@GuardSatisfied LineNumberReader this, @NonNegative int lineNumber);

    @Positive
    @NonNegative
    @Positive
    public int getLineNumber(@GuardSatisfied LineNumberReader this);

    @Positive
    @SuppressWarnings("fallthrough")
    @Positive
    public int read(@GuardSatisfied LineNumberReader this) throws IOException;

    @Positive
    @SuppressWarnings("fallthrough")
    @Positive
    @GTENegativeOne
    @Positive
    @LTEqLengthOf({ "#1" })
    @Positive
    public int read(@GuardSatisfied LineNumberReader this, char[] cbuf, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len) throws IOException;

    @Positive
    @Nullable
    @Positive
    public String readLine(@GuardSatisfied LineNumberReader this) throws IOException;

    @Positive
    @NonNegative
    @Positive
    public long skip(@GuardSatisfied LineNumberReader this, @NonNegative long n) throws IOException;

    @Positive
    public void mark(@GuardSatisfied LineNumberReader this, @NonNegative int readAheadLimit) throws IOException;

    @Positive
    public void reset(@GuardSatisfied LineNumberReader this) throws IOException;
    @Positive
}
