/*
    @Positive
 * Copyright (c) 1996, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.text;

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
import java.io.InvalidObjectException;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.util.Arrays;

    @Positive
public class ChoiceFormat extends NumberFormat {

    @Positive
    public void applyPattern(String newPattern);

    @Positive
    public String toPattern();

    @Positive
    public ChoiceFormat(String newPattern) {
    @Positive
    }

    @Positive
    public ChoiceFormat(double[] limits, String[] formats) {
    @Positive
    }

    @Positive
    public void setChoices(double[] limits, String[] formats);

    @Positive
    public double[] getLimits();

    @Positive
    public Object[] getFormats();

    @Positive
    public StringBuffer format(long number, StringBuffer toAppendTo, FieldPosition status);

    @Positive
    public StringBuffer format(double number, StringBuffer toAppendTo, FieldPosition status);

    @Positive
    public Number parse(String text, ParsePosition status);

    @Positive
    public static final double nextDouble(double d);

    @Positive
    public static final double previousDouble(double d);

    @Positive
    public Object clone();

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public static double nextDouble(double d, boolean positive);
    @Positive
}

// CFWR semantic augmentation - variant 0
