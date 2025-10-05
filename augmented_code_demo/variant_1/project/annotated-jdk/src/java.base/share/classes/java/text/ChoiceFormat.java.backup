/*
 * Copyright (c) 1996, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.text;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.InvalidObjectException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Arrays;

public class ChoiceFormat extends NumberFormat {

    public void applyPattern(String newPattern);

    public String toPattern();

    public ChoiceFormat(String newPattern) {
    }

    public ChoiceFormat(double[] limits, String[] formats) {
    }

    public void setChoices(double[] limits, String[] formats);

    public double[] getLimits();

    public Object[] getFormats();

    public StringBuffer format(long number, StringBuffer toAppendTo, FieldPosition status);

    public StringBuffer format(double number, StringBuffer toAppendTo, FieldPosition status);

    public Number parse(String text, ParsePosition status);

    public static final double nextDouble(double d);

    public static final double previousDouble(double d);

    public Object clone();

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public static double nextDouble(double d, boolean positive);
}
