/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2012, 2013, Oracle and/or its affiliates. All rights reserved.
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
package java.time.format;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.text.DecimalFormatSymbols;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.concurrent.ConcurrentMap;

    @Positive
public final class DecimalStyle {

    @Positive
    public static final DecimalStyle STANDARD;

    @Positive
    public static Set<Locale> getAvailableLocales();

    @Positive
    public static DecimalStyle ofDefaultLocale();

    @Positive
    public static DecimalStyle of(Locale locale);

    @Positive
    public char getZeroDigit();

    @Positive
    public DecimalStyle withZeroDigit(char zeroDigit);

    @Positive
    public char getPositiveSign();

    @Positive
    public DecimalStyle withPositiveSign(char positiveSign);

    @Positive
    public char getNegativeSign();

    @Positive
    public DecimalStyle withNegativeSign(char negativeSign);

    @Positive
    public char getDecimalSeparator();

    @Positive
    public DecimalStyle withDecimalSeparator(char decimalSeparator);

    @Positive
    int convertToDigit(char ch);

    @Positive
    String convertNumberToI18N(String numericText);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    public String toString();
    @Positive
}
