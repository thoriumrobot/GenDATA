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
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.text.spi.NumberFormatProvider;
import java.util.ArrayList;
import java.util.Currency;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import sun.util.locale.provider.LocaleProviderAdapter;
import sun.util.locale.provider.ResourceBundleBasedAdapter;

public class DecimalFormat extends NumberFormat {

    public DecimalFormat() {
    }

    public DecimalFormat(String pattern) {
    }

    public DecimalFormat(String pattern, DecimalFormatSymbols symbols) {
    }

    @Override
    public final StringBuffer format(Object number, StringBuffer toAppendTo, FieldPosition pos);

    @Override
    public StringBuffer format(double number, StringBuffer result, FieldPosition fieldPosition);

    StringBuffer format(double number, StringBuffer result, FieldDelegate delegate);

    boolean handleNaN(double number, StringBuffer result, FieldDelegate delegate);

    boolean handleInfinity(double number, StringBuffer result, FieldDelegate delegate, boolean isNegative);

    StringBuffer doubleSubformat(double number, StringBuffer result, FieldDelegate delegate, boolean isNegative);

    @Override
    public StringBuffer format(long number, StringBuffer result, FieldPosition fieldPosition);

    StringBuffer format(long number, StringBuffer result, FieldDelegate delegate);

    StringBuffer format(BigDecimal number, StringBuffer result, FieldDelegate delegate);

    StringBuffer format(BigInteger number, StringBuffer result, FieldDelegate delegate, boolean formatLong);

    @Override
    public AttributedCharacterIterator formatToCharacterIterator(Object obj);

    String fastFormat(double d);

    void setDigitList(Number number, boolean isNegative, int maxDigits);

    void subformatNumber(StringBuffer result, FieldDelegate delegate, boolean isNegative, boolean isInteger, int maxIntDigits, int minIntDigits, int maxFraDigits, int minFraDigits);

    @Override
    public Number parse(String text, ParsePosition pos);

    int subparseNumber(String text, int position, DigitList digits, boolean checkExponent, boolean isExponent, boolean[] status);

    public DecimalFormatSymbols getDecimalFormatSymbols();

    public void setDecimalFormatSymbols(DecimalFormatSymbols newSymbols);

    public String getPositivePrefix();

    public void setPositivePrefix(String newValue);

    public String getNegativePrefix();

    public void setNegativePrefix(String newValue);

    public String getPositiveSuffix();

    public void setPositiveSuffix(String newValue);

    public String getNegativeSuffix();

    public void setNegativeSuffix(String newValue);

    public int getMultiplier();

    public void setMultiplier(int newValue);

    @Override
    public void setGroupingUsed(boolean newValue);

    public int getGroupingSize();

    public void setGroupingSize(int newValue);

    public boolean isDecimalSeparatorAlwaysShown();

    public void setDecimalSeparatorAlwaysShown(boolean newValue);

    public boolean isParseBigDecimal();

    public void setParseBigDecimal(boolean newValue);

    @Override
    public Object clone();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    public String toPattern();

    public String toLocalizedPattern();

    public void applyPattern(String pattern);

    public void applyLocalizedPattern(String pattern);

    @Override
    public void setMaximumIntegerDigits(int newValue);

    @Override
    public void setMinimumIntegerDigits(int newValue);

    @Override
    public void setMaximumFractionDigits(int newValue);

    @Override
    public void setMinimumFractionDigits(int newValue);

    @Override
    public int getMaximumIntegerDigits();

    @Override
    public int getMinimumIntegerDigits();

    @Override
    public int getMaximumFractionDigits();

    @Override
    public int getMinimumFractionDigits();

    @Override
    public Currency getCurrency();

    @Override
    public void setCurrency(Currency currency);

    @Override
    public RoundingMode getRoundingMode();

    @Override
    public void setRoundingMode(RoundingMode roundingMode);

    private static class FastPathData {
    }

    private static class DigitArrays {
    }
}
