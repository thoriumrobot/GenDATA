/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.common.value.qual.IntRange;
    @Positive
import org.checkerframework.common.value.qual.IntVal;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.Arrays;

    @Positive
@AnnotatedFor({ "index", "interning", "lock", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public class StreamTokenizer {

    @Positive
    @IntRange(from = -4, to = 65535)
    @Positive
    public int ttype;

    @Positive
    @IntVal(-1)
    @Positive
    public static final int TT_EOF;

    @Positive
    @IntVal('\n')
    @Positive
    public static final int TT_EOL;

    @Positive
    @IntVal(-2)
    @Positive
    public static final int TT_NUMBER;

    @Positive
    @IntVal(-3)
    @Positive
    public static final int TT_WORD;

    @Positive
    @Nullable
    @Positive
    public String sval;

    @Positive
    public double nval;

    @Positive
    @Deprecated
    @Positive
    public StreamTokenizer(InputStream is) {
    @Positive
    }

    @Positive
    public StreamTokenizer(Reader r) {
    @Positive
    }

    @Positive
    public void resetSyntax();

    @Positive
    public void wordChars(int low, int hi);

    @Positive
    public void whitespaceChars(int low, int hi);

    @Positive
    public void ordinaryChars(int low, int hi);

    @Positive
    public void ordinaryChar(int ch);

    @Positive
    public void commentChar(int ch);

    @Positive
    public void quoteChar(int ch);

    @Positive
    public void parseNumbers();

    @Positive
    public void eolIsSignificant(boolean flag);

    @Positive
    public void slashStarComments(boolean flag);

    @Positive
    public void slashSlashComments(boolean flag);

    @Positive
    public void lowerCaseMode(boolean fl);

    @Positive
    public int nextToken() throws IOException;

    @Positive
    public void pushBack();

    @Positive
    @NonNegative
    @Positive
    public int lineno();

    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied StreamTokenizer this);
    @Positive
}
