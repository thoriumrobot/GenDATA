/*
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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
import java.io.ObjectOutputStream;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.text.spi.NumberFormatProvider;
import java.util.Currency;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import sun.util.locale.provider.LocaleProviderAdapter;
import sun.util.locale.provider.LocaleServiceProviderPool;

public abstract class NumberFormat extends Format {

    public static final int INTEGER_FIELD;

    public static final int FRACTION_FIELD;

    protected NumberFormat() {
    }

    @Override
    public StringBuffer format(Object number, StringBuffer toAppendTo, FieldPosition pos);

    @Override
    public final Object parseObject(String source, ParsePosition pos);

    public final String format(double number);

    String fastFormat(double number);

    public final String format(long number);

    public abstract StringBuffer format(double number, StringBuffer toAppendTo, FieldPosition pos);

    public abstract StringBuffer format(long number, StringBuffer toAppendTo, FieldPosition pos);

    public abstract Number parse(String source, ParsePosition parsePosition);

    public Number parse(String source) throws ParseException;

    public boolean isParseIntegerOnly();

    public void setParseIntegerOnly(boolean value);

    public static final NumberFormat getInstance();

    public static NumberFormat getInstance(Locale inLocale);

    public static final NumberFormat getNumberInstance();

    public static NumberFormat getNumberInstance(Locale inLocale);

    public static final NumberFormat getIntegerInstance();

    public static NumberFormat getIntegerInstance(Locale inLocale);

    public static final NumberFormat getCurrencyInstance();

    public static NumberFormat getCurrencyInstance(Locale inLocale);

    public static final NumberFormat getPercentInstance();

    public static NumberFormat getPercentInstance(Locale inLocale);

    static final NumberFormat getScientificInstance();

    static NumberFormat getScientificInstance(Locale inLocale);

    public static NumberFormat getCompactNumberInstance();

    public static NumberFormat getCompactNumberInstance(Locale locale, NumberFormat.Style formatStyle);

    public static Locale[] getAvailableLocales();

    @Override
    public int hashCode();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public Object clone();

    public boolean isGroupingUsed();

    public void setGroupingUsed(boolean newValue);

    public int getMaximumIntegerDigits();

    public void setMaximumIntegerDigits(int newValue);

    public int getMinimumIntegerDigits();

    public void setMinimumIntegerDigits(int newValue);

    public int getMaximumFractionDigits();

    public void setMaximumFractionDigits(int newValue);

    public int getMinimumFractionDigits();

    public void setMinimumFractionDigits(int newValue);

    public Currency getCurrency();

    public void setCurrency(Currency currency);

    public RoundingMode getRoundingMode();

    public void setRoundingMode(RoundingMode roundingMode);

    public static class Field extends Format.Field {

        protected Field(String name) {
        }

        @Override
        @java.io.Serial
        protected Object readResolve() throws InvalidObjectException;

        public static final Field INTEGER;

        public static final Field FRACTION;

        public static final Field EXPONENT;

        public static final Field DECIMAL_SEPARATOR;

        public static final Field SIGN;

        public static final Field GROUPING_SEPARATOR;

        public static final Field EXPONENT_SYMBOL;

        public static final Field PERCENT;

        public static final Field PERMILLE;

        public static final Field CURRENCY;

        public static final Field EXPONENT_SIGN;

        public static final Field PREFIX;

        public static final Field SUFFIX;
    }

    public enum Style {

        SHORT, LONG
    }
}
