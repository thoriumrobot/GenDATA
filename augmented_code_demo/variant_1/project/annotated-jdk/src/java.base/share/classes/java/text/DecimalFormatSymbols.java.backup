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
import java.io.Serializable;
import java.text.spi.DecimalFormatSymbolsProvider;
import java.util.Currency;
import java.util.Locale;
import java.util.Objects;
import sun.util.locale.provider.CalendarDataUtility;
import sun.util.locale.provider.LocaleProviderAdapter;
import sun.util.locale.provider.LocaleServiceProviderPool;
import sun.util.locale.provider.ResourceBundleBasedAdapter;

public class DecimalFormatSymbols implements Cloneable, Serializable {

    public DecimalFormatSymbols() {
    }

    public DecimalFormatSymbols(Locale locale) {
    }

    public static Locale[] getAvailableLocales();

    public static final DecimalFormatSymbols getInstance();

    public static final DecimalFormatSymbols getInstance(Locale locale);

    public char getZeroDigit();

    public void setZeroDigit(char zeroDigit);

    public char getGroupingSeparator();

    public void setGroupingSeparator(char groupingSeparator);

    public char getDecimalSeparator();

    public void setDecimalSeparator(char decimalSeparator);

    public char getPerMill();

    public void setPerMill(char perMill);

    public char getPercent();

    public void setPercent(char percent);

    public char getDigit();

    public void setDigit(char digit);

    public char getPatternSeparator();

    public void setPatternSeparator(char patternSeparator);

    public String getInfinity();

    public void setInfinity(String infinity);

    public String getNaN();

    public void setNaN(String NaN);

    public char getMinusSign();

    public void setMinusSign(char minusSign);

    public String getCurrencySymbol();

    public void setCurrencySymbol(String currency);

    public String getInternationalCurrencySymbol();

    public void setInternationalCurrencySymbol(String currencyCode);

    public Currency getCurrency();

    public void setCurrency(Currency currency);

    public char getMonetaryDecimalSeparator();

    public void setMonetaryDecimalSeparator(char sep);

    public String getExponentSeparator();

    public void setExponentSeparator(String exp);

    public char getMonetaryGroupingSeparator();

    public void setMonetaryGroupingSeparator(char monetaryGroupingSeparator);

    char getExponentialSymbol();

    void setExponentialSymbol(char exp);

    String getPerMillText();

    void setPerMillText(String perMillText);

    String getPercentText();

    void setPercentText(String percentText);

    String getMinusSignText();

    void setMinusSignText(String minusSignText);

    @Override
    public Object clone();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();
}
