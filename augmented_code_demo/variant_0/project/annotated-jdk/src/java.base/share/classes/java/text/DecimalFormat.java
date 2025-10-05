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
import java.io.IOException;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.math.BigDecimal;
    @Positive
import java.math.BigInteger;
    @Positive
import java.math.RoundingMode;
    @Positive
import java.text.spi.NumberFormatProvider;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Currency;
    @Positive
import java.util.Locale;
    @Positive
import java.util.concurrent.atomic.AtomicInteger;
    @Positive
import java.util.concurrent.atomic.AtomicLong;
    @Positive
import sun.util.locale.provider.LocaleProviderAdapter;
    @Positive
import sun.util.locale.provider.ResourceBundleBasedAdapter;

    @Positive
public class DecimalFormat extends NumberFormat {

    @Positive
    public DecimalFormat() {
    @Positive
    }

    @Positive
    public DecimalFormat(String pattern) {
    @Positive
    }

    @Positive
    public DecimalFormat(String pattern, DecimalFormatSymbols symbols) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public final StringBuffer format(Object number, StringBuffer toAppendTo, FieldPosition pos);

    @Positive
    @Override
    @Positive
    public StringBuffer format(double number, StringBuffer result, FieldPosition fieldPosition);

    @Positive
    StringBuffer format(double number, StringBuffer result, FieldDelegate delegate);

    @Positive
    boolean handleNaN(double number, StringBuffer result, FieldDelegate delegate);

    @Positive
    boolean handleInfinity(double number, StringBuffer result, FieldDelegate delegate, boolean isNegative);

    @Positive
    StringBuffer doubleSubformat(double number, StringBuffer result, FieldDelegate delegate, boolean isNegative);

    @Positive
    @Override
    @Positive
    public StringBuffer format(long number, StringBuffer result, FieldPosition fieldPosition);

    @Positive
    StringBuffer format(long number, StringBuffer result, FieldDelegate delegate);

    @Positive
    StringBuffer format(BigDecimal number, StringBuffer result, FieldDelegate delegate);

    @Positive
    StringBuffer format(BigInteger number, StringBuffer result, FieldDelegate delegate, boolean formatLong);

    @Positive
    @Override
    @Positive
    public AttributedCharacterIterator formatToCharacterIterator(Object obj);

    @Positive
    String fastFormat(double d);

    @Positive
    void setDigitList(Number number, boolean isNegative, int maxDigits);

    @Positive
    void subformatNumber(StringBuffer result, FieldDelegate delegate, boolean isNegative, boolean isInteger, int maxIntDigits, int minIntDigits, int maxFraDigits, int minFraDigits);

    @Positive
    @Override
    @Positive
    public Number parse(String text, ParsePosition pos);

    @Positive
    int subparseNumber(String text, int position, DigitList digits, boolean checkExponent, boolean isExponent, boolean[] status);

    @Positive
    public DecimalFormatSymbols getDecimalFormatSymbols();

    @Positive
    public void setDecimalFormatSymbols(DecimalFormatSymbols newSymbols);

    @Positive
    public String getPositivePrefix();

    @Positive
    public void setPositivePrefix(String newValue);

    @Positive
    public String getNegativePrefix();

    @Positive
    public void setNegativePrefix(String newValue);

    @Positive
    public String getPositiveSuffix();

    @Positive
    public void setPositiveSuffix(String newValue);

    @Positive
    public String getNegativeSuffix();

    @Positive
    public void setNegativeSuffix(String newValue);

    @Positive
    public int getMultiplier();

    @Positive
    public void setMultiplier(int newValue);

    @Positive
    @Override
    @Positive
    public void setGroupingUsed(boolean newValue);

    @Positive
    public int getGroupingSize();

    @Positive
    public void setGroupingSize(int newValue);

    @Positive
    public boolean isDecimalSeparatorAlwaysShown();

    @Positive
    public void setDecimalSeparatorAlwaysShown(boolean newValue);

    @Positive
    public boolean isParseBigDecimal();

    @Positive
    public void setParseBigDecimal(boolean newValue);

    @Positive
    @Override
    @Positive
    public Object clone();

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
    public String toPattern();

    @Positive
    public String toLocalizedPattern();

    @Positive
    public void applyPattern(String pattern);

    @Positive
    public void applyLocalizedPattern(String pattern);

    @Positive
    @Override
    @Positive
    public void setMaximumIntegerDigits(int newValue);

    @Positive
    @Override
    @Positive
    public void setMinimumIntegerDigits(int newValue);

    @Positive
    @Override
    @Positive
    public void setMaximumFractionDigits(int newValue);

    @Positive
    @Override
    @Positive
    public void setMinimumFractionDigits(int newValue);

    @Positive
    @Override
    @Positive
    public int getMaximumIntegerDigits();

    @Positive
    @Override
    @Positive
    public int getMinimumIntegerDigits();

    @Positive
    @Override
    @Positive
    public int getMaximumFractionDigits();

    @Positive
    @Override
    @Positive
    public int getMinimumFractionDigits();

    @Positive
    @Override
    @Positive
    public Currency getCurrency();

    @Positive
    @Override
    @Positive
    public void setCurrency(Currency currency);

    @Positive
    @Override
    @Positive
    public RoundingMode getRoundingMode();

    @Positive
    @Override
    @Positive
    public void setRoundingMode(RoundingMode roundingMode);

    @Positive
    private static class FastPathData {
    @Positive
    }

    @Positive
    private static class DigitArrays {
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
