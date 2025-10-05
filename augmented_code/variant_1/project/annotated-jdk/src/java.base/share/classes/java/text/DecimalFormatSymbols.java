/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
import java.io.Serializable;
    @Positive
import java.text.spi.DecimalFormatSymbolsProvider;
    @Positive
import java.util.Currency;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Objects;
    @Positive
import sun.util.locale.provider.CalendarDataUtility;
    @Positive
import sun.util.locale.provider.LocaleProviderAdapter;
    @Positive
import sun.util.locale.provider.LocaleServiceProviderPool;
    @Positive
import sun.util.locale.provider.ResourceBundleBasedAdapter;

    @Positive
public class DecimalFormatSymbols implements Cloneable, Serializable {

    @Positive
    public DecimalFormatSymbols() {
    @Positive
    }

    @Positive
    public DecimalFormatSymbols(Locale locale) {
    @Positive
    }

    @Positive
    public static Locale[] getAvailableLocales();

    @Positive
    public static final DecimalFormatSymbols getInstance();

    @Positive
    public static final DecimalFormatSymbols getInstance(Locale locale);

    @Positive
    public char getZeroDigit();

    @Positive
    public void setZeroDigit(char zeroDigit);

    @Positive
    public char getGroupingSeparator();

    @Positive
    public void setGroupingSeparator(char groupingSeparator);

    @Positive
    public char getDecimalSeparator();

    @Positive
    public void setDecimalSeparator(char decimalSeparator);

    @Positive
    public char getPerMill();

    @Positive
    public void setPerMill(char perMill);

    @Positive
    public char getPercent();

    @Positive
    public void setPercent(char percent);

    @Positive
    public char getDigit();

    @Positive
    public void setDigit(char digit);

    @Positive
    public char getPatternSeparator();

    @Positive
    public void setPatternSeparator(char patternSeparator);

    @Positive
    public String getInfinity();

    @Positive
    public void setInfinity(String infinity);

    @Positive
    public String getNaN();

    @Positive
    public void setNaN(String NaN);

    @Positive
    public char getMinusSign();

    @Positive
    public void setMinusSign(char minusSign);

    @Positive
    public String getCurrencySymbol();

    @Positive
    public void setCurrencySymbol(String currency);

    @Positive
    public String getInternationalCurrencySymbol();

    @Positive
    public void setInternationalCurrencySymbol(String currencyCode);

    @Positive
    public Currency getCurrency();

    @Positive
    public void setCurrency(Currency currency);

    @Positive
    public char getMonetaryDecimalSeparator();

    @Positive
    public void setMonetaryDecimalSeparator(char sep);

    @Positive
    public String getExponentSeparator();

    @Positive
    public void setExponentSeparator(String exp);

    @Positive
    public char getMonetaryGroupingSeparator();

    @Positive
    public void setMonetaryGroupingSeparator(char monetaryGroupingSeparator);

    @Positive
    char getExponentialSymbol();

    @Positive
    void setExponentialSymbol(char exp);

    @Positive
    String getPerMillText();

    @Positive
    void setPerMillText(String perMillText);

    @Positive
    String getPercentText();

    @Positive
    void setPercentText(String percentText);

    @Positive
    String getMinusSignText();

    @Positive
    void setMinusSignText(String minusSignText);

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
}
