/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1999, 2000, Oracle and/or its affiliates. All rights reserved.
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
package java.util.regex;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;

    @Positive
@AnnotatedFor({ "index", "interning" })
    @Positive
@UsesObjectEquals
    @Positive
final class ASCII {

    @Positive
    static int getType(int ch);

    @Positive
    static boolean isType(int ch, int type);

    @Positive
    static boolean isAscii(int ch);

    @Positive
    static boolean isAlpha(int ch);

    @Positive
    static boolean isDigit(int ch);

    @Positive
    static boolean isAlnum(int ch);

    @Positive
    static boolean isGraph(int ch);

    @Positive
    static boolean isPrint(int ch);

    @Positive
    static boolean isPunct(int ch);

    @Positive
    static boolean isSpace(int ch);

    @Positive
    static boolean isHexDigit(int ch);

    @Positive
    static boolean isOctDigit(int ch);

    @Positive
    static boolean isCntrl(int ch);

    @Positive
    static boolean isLower(int ch);

    @Positive
    static boolean isUpper(int ch);

    @Positive
    static boolean isWord(int ch);

    @Positive
    static int toDigit(int ch);

    @Positive
    static int toLower(int ch);

    @Positive
    static int toUpper(int ch);
    @Positive
}
