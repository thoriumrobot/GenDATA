/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2006, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.common.value.qual.IntRange;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;

    @Positive
@AnnotatedFor({ "index", "interning" })
    @Positive
@UsesObjectEquals
    @Positive
abstract class CharacterData {

    @Positive
    abstract int getProperties(int ch);

    @Positive
    abstract int getType(int ch);

    @Positive
    abstract boolean isDigit(int ch);

    @Positive
    abstract boolean isLowerCase(int ch);

    @Positive
    abstract boolean isUpperCase(int ch);

    @Positive
    abstract boolean isWhitespace(int ch);

    @Positive
    abstract boolean isMirrored(int ch);

    @Positive
    abstract boolean isJavaIdentifierStart(int ch);

    @Positive
    abstract boolean isJavaIdentifierPart(int ch);

    @Positive
    abstract boolean isUnicodeIdentifierStart(int ch);

    @Positive
    abstract boolean isUnicodeIdentifierPart(int ch);

    @Positive
    abstract boolean isIdentifierIgnorable(int ch);

    @Positive
    abstract int toLowerCase(int ch);

    @Positive
    abstract int toUpperCase(int ch);

    @Positive
    abstract int toTitleCase(int ch);

    @Positive
    abstract int digit(int ch, @IntRange(from = 2, to = 36) int radix);

    @Positive
    abstract int getNumericValue(int ch);

    @Positive
    abstract byte getDirectionality(int ch);

    @Positive
    int toUpperCaseEx(int ch);

    @Positive
    char[] toUpperCaseCharArray(int ch);

    @Positive
    boolean isOtherAlphabetic(int ch);

    @Positive
    boolean isIdeographic(int ch);

    @Positive
    static final CharacterData of(int ch);
    @Positive
}
