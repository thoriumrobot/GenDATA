/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2006, 2020, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package java.sql;

    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import java.util.Map;

    @Positive
public class SQLClientInfoException extends SQLException {

    @Positive
    public SQLClientInfoException() {
    @Positive
    }

    @Positive
    public SQLClientInfoException(@Nullable Map<String, ClientInfoStatus> failedProperties) {
    @Positive
    }

    @Positive
    public SQLClientInfoException(@Nullable Map<String, ClientInfoStatus> failedProperties, @Nullable Throwable cause) {
    @Positive
    }

    @Positive
    public SQLClientInfoException(@Nullable String reason, @Nullable Map<String, ClientInfoStatus> failedProperties) {
    @Positive
    }

    @Positive
    public SQLClientInfoException(@Nullable String reason, @Nullable Map<String, ClientInfoStatus> failedProperties, @Nullable Throwable cause) {
    @Positive
    }

    @Positive
    public SQLClientInfoException(@Nullable String reason, @Nullable String SQLState, @Nullable Map<String, ClientInfoStatus> failedProperties) {
    @Positive
    }

    @Positive
    public SQLClientInfoException(@Nullable String reason, @Nullable String SQLState, @Nullable Map<String, ClientInfoStatus> failedProperties, @Nullable Throwable cause) {
    @Positive
    }

    @Positive
    public SQLClientInfoException(@Nullable String reason, @Nullable String SQLState, int vendorCode, @Nullable Map<String, ClientInfoStatus> failedProperties) {
    @Positive
    }

    @Positive
    public SQLClientInfoException(@Nullable String reason, @Nullable String SQLState, int vendorCode, @Nullable Map<String, ClientInfoStatus> failedProperties, @Nullable Throwable cause) {
    @Positive
    }

    @Positive
    @Nullable
    @Positive
    public Map<String, ClientInfoStatus> getFailedProperties();
    @Positive
}
