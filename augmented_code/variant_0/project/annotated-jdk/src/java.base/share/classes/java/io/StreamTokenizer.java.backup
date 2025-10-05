/*
 * Copyright (c) 1995, 2020, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.common.value.qual.IntRange;
import org.checkerframework.common.value.qual.IntVal;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.Arrays;

@AnnotatedFor({ "index", "interning", "lock", "nullness" })
@UsesObjectEquals
public class StreamTokenizer {

    @IntRange(from = -4, to = 65535)
    public int ttype;

    @IntVal(-1)
    public static final int TT_EOF;

    @IntVal('\n')
    public static final int TT_EOL;

    @IntVal(-2)
    public static final int TT_NUMBER;

    @IntVal(-3)
    public static final int TT_WORD;

    @Nullable
    public String sval;

    public double nval;

    @Deprecated
    public StreamTokenizer(InputStream is) {
    }

    public StreamTokenizer(Reader r) {
    }

    public void resetSyntax();

    public void wordChars(int low, int hi);

    public void whitespaceChars(int low, int hi);

    public void ordinaryChars(int low, int hi);

    public void ordinaryChar(int ch);

    public void commentChar(int ch);

    public void quoteChar(int ch);

    public void parseNumbers();

    public void eolIsSignificant(boolean flag);

    public void slashStarComments(boolean flag);

    public void slashSlashComments(boolean flag);

    public void lowerCaseMode(boolean fl);

    public int nextToken() throws IOException;

    public void pushBack();

    @NonNegative
    public int lineno();

    @SideEffectFree
    public String toString(@GuardSatisfied StreamTokenizer this);
}
