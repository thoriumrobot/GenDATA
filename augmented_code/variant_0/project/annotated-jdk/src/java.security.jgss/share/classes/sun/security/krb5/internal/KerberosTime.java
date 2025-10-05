/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
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
package sun.security.krb5.internal;

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
import sun.security.krb5.Asn1Exception;
    @Positive
import sun.security.krb5.Config;
    @Positive
import sun.security.krb5.KrbException;
    @Positive
import sun.security.util.DerInputStream;
    @Positive
import sun.security.util.DerOutputStream;
    @Positive
import sun.security.util.DerValue;
    @Positive
import java.io.IOException;
    @Positive
import java.time.Instant;
    @Positive
import java.util.Calendar;
    @Positive
import java.util.Date;
    @Positive
import java.util.TimeZone;

    @Positive
public class KerberosTime {

    @Positive
    public KerberosTime(long time) {
    @Positive
    }

    @Positive
    public KerberosTime(String time) throws Asn1Exception {
    @Positive
    }

    @Positive
    public KerberosTime(Date time) {
    @Positive
    }

    @Positive
    public KerberosTime(Instant instant) {
    @Positive
    }

    @Positive
    public static KerberosTime now();

    @Positive
    public String toGeneralizedTimeString();

    @Positive
    public byte[] asn1Encode() throws Asn1Exception, IOException;

    @Positive
    public long getTime();

    @Positive
    public Date toDate();

    @Positive
    public int getMicroSeconds();

    @Positive
    public KerberosTime withMicroSeconds(int usec);

    @Positive
    public boolean inClockSkew();

    @Positive
    public boolean greaterThanWRTClockSkew(KerberosTime time, int clockSkew);

    @Positive
    public boolean greaterThanWRTClockSkew(KerberosTime time);

    @Positive
    public boolean greaterThan(KerberosTime time);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public boolean isZero();

    @Positive
    public int getSeconds();

    @Positive
    public static KerberosTime parse(DerInputStream data, byte explicitTag, boolean optional) throws Asn1Exception, IOException;

    @Positive
    public static int getDefaultSkew();

    @Positive
    public String toString();
    @Positive
}
