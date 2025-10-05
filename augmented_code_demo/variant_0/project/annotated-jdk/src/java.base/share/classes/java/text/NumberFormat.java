/*
    @Positive
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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
import java.io.ObjectOutputStream;
    @Positive
import java.math.BigInteger;
    @Positive
import java.math.RoundingMode;
    @Positive
import java.text.spi.NumberFormatProvider;
    @Positive
import java.util.Currency;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.concurrent.atomic.AtomicInteger;
    @Positive
import java.util.concurrent.atomic.AtomicLong;
    @Positive
import sun.util.locale.provider.LocaleProviderAdapter;
    @Positive
import sun.util.locale.provider.LocaleServiceProviderPool;

    @Positive
public abstract class NumberFormat extends Format {

    @Positive
    public static final int INTEGER_FIELD;

    @Positive
    public static final int FRACTION_FIELD;

    @Positive
    protected NumberFormat() {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public StringBuffer format(Object number, StringBuffer toAppendTo, FieldPosition pos);

    @Positive
    @Override
    @Positive
    public final Object parseObject(String source, ParsePosition pos);

    @Positive
    public final String format(double number);

    @Positive
    String fastFormat(double number);

    @Positive
    public final String format(long number);

    @Positive
    public abstract StringBuffer format(double number, StringBuffer toAppendTo, FieldPosition pos);

    @Positive
    public abstract StringBuffer format(long number, StringBuffer toAppendTo, FieldPosition pos);

    @Positive
    public abstract Number parse(String source, ParsePosition parsePosition);

    @Positive
    public Number parse(String source) throws ParseException;

    @Positive
    public boolean isParseIntegerOnly();

    @Positive
    public void setParseIntegerOnly(boolean value);

    @Positive
    public static final NumberFormat getInstance();

    @Positive
    public static NumberFormat getInstance(Locale inLocale);

    @Positive
    public static final NumberFormat getNumberInstance();

    @Positive
    public static NumberFormat getNumberInstance(Locale inLocale);

    @Positive
    public static final NumberFormat getIntegerInstance();

    @Positive
    public static NumberFormat getIntegerInstance(Locale inLocale);

    @Positive
    public static final NumberFormat getCurrencyInstance();

    @Positive
    public static NumberFormat getCurrencyInstance(Locale inLocale);

    @Positive
    public static final NumberFormat getPercentInstance();

    @Positive
    public static NumberFormat getPercentInstance(Locale inLocale);

    @Positive
    static final NumberFormat getScientificInstance();

    @Positive
    static NumberFormat getScientificInstance(Locale inLocale);

    @Positive
    public static NumberFormat getCompactNumberInstance();

    @Positive
    public static NumberFormat getCompactNumberInstance(Locale locale, NumberFormat.Style formatStyle);

    @Positive
    public static Locale[] getAvailableLocales();

    @Positive
    @Override
    @Positive
    public int hashCode();

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
    public Object clone();

    @Positive
    public boolean isGroupingUsed();

    @Positive
    public void setGroupingUsed(boolean newValue);

    @Positive
    public int getMaximumIntegerDigits();

    @Positive
    public void setMaximumIntegerDigits(int newValue);

    @Positive
    public int getMinimumIntegerDigits();

    @Positive
    public void setMinimumIntegerDigits(int newValue);

    @Positive
    public int getMaximumFractionDigits();

    @Positive
    public void setMaximumFractionDigits(int newValue);

    @Positive
    public int getMinimumFractionDigits();

    @Positive
    public void setMinimumFractionDigits(int newValue);

    @Positive
    public Currency getCurrency();

    @Positive
    public void setCurrency(Currency currency);

    @Positive
    public RoundingMode getRoundingMode();

    @Positive
    public void setRoundingMode(RoundingMode roundingMode);

    @Positive
    public static class Field extends Format.Field {

    @Positive
        protected Field(String name) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        @java.io.Serial
    @Positive
        protected Object readResolve() throws InvalidObjectException;

    @Positive
        public static final Field INTEGER;

    @Positive
        public static final Field FRACTION;

    @Positive
        public static final Field EXPONENT;

    @Positive
        public static final Field DECIMAL_SEPARATOR;

    @Positive
        public static final Field SIGN;

    @Positive
        public static final Field GROUPING_SEPARATOR;

    @Positive
        public static final Field EXPONENT_SYMBOL;

    @Positive
        public static final Field PERCENT;

    @Positive
        public static final Field PERMILLE;

    @Positive
        public static final Field CURRENCY;

    @Positive
        public static final Field EXPONENT_SIGN;

    @Positive
        public static final Field PREFIX;

    @Positive
        public static final Field SUFFIX;
    @Positive
    }

    @Positive
    public enum Style {

    @Positive
        SHORT, LONG
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
