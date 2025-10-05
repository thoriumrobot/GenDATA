/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
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
package javax.xml.datatype;

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
import javax.xml.namespace.QName;
    @Positive
import java.math.BigDecimal;
    @Positive
import java.math.BigInteger;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.TimeZone;
    @Positive
import java.util.GregorianCalendar;

    @Positive
public abstract class XMLGregorianCalendar implements Cloneable {

    @Positive
    public XMLGregorianCalendar() {
    @Positive
    }

    @Positive
    public abstract void clear();

    @Positive
    public abstract void reset();

    @Positive
    public abstract void setYear(BigInteger year);

    @Positive
    public abstract void setYear(int year);

    @Positive
    public abstract void setMonth(int month);

    @Positive
    public abstract void setDay(int day);

    @Positive
    public abstract void setTimezone(int offset);

    @Positive
    public void setTime(int hour, int minute, int second);

    @Positive
    public abstract void setHour(int hour);

    @Positive
    public abstract void setMinute(int minute);

    @Positive
    public abstract void setSecond(int second);

    @Positive
    public abstract void setMillisecond(int millisecond);

    @Positive
    public abstract void setFractionalSecond(BigDecimal fractional);

    @Positive
    public void setTime(int hour, int minute, int second, BigDecimal fractional);

    @Positive
    public void setTime(int hour, int minute, int second, int millisecond);

    @Positive
    public abstract BigInteger getEon();

    @Positive
    public abstract int getYear();

    @Positive
    public abstract BigInteger getEonAndYear();

    @Positive
    public abstract int getMonth();

    @Positive
    public abstract int getDay();

    @Positive
    public abstract int getTimezone();

    @Positive
    public abstract int getHour();

    @Positive
    public abstract int getMinute();

    @Positive
    public abstract int getSecond();

    @Positive
    public int getMillisecond();

    @Positive
    public abstract BigDecimal getFractionalSecond();

    @Positive
    public abstract int compare(XMLGregorianCalendar xmlGregorianCalendar);

    @Positive
    public abstract XMLGregorianCalendar normalize();

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
    public abstract String toXMLFormat();

    @Positive
    public abstract QName getXMLSchemaType();

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    public abstract boolean isValid();

    @Positive
    public abstract void add(Duration duration);

    @Positive
    public abstract GregorianCalendar toGregorianCalendar();

    @Positive
    public abstract GregorianCalendar toGregorianCalendar(java.util.TimeZone timezone, java.util.Locale aLocale, XMLGregorianCalendar defaults);

    @Positive
    public abstract TimeZone getTimeZone(int defaultZoneoffset);

    @Positive
    @Override
    @Positive
    public abstract Object clone();
    @Positive
}
