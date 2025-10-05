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
package java.sql;

import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;

@AnnotatedFor("nullness")
public interface DatabaseMetaData extends Wrapper {

    boolean allProceduresAreCallable() throws SQLException;

    boolean allTablesAreSelectable() throws SQLException;

    @Nullable
    String getURL() throws SQLException;

    String getUserName() throws SQLException;

    boolean isReadOnly() throws SQLException;

    boolean nullsAreSortedHigh() throws SQLException;

    boolean nullsAreSortedLow() throws SQLException;

    boolean nullsAreSortedAtStart() throws SQLException;

    boolean nullsAreSortedAtEnd() throws SQLException;

    String getDatabaseProductName() throws SQLException;

    String getDatabaseProductVersion() throws SQLException;

    String getDriverName() throws SQLException;

    String getDriverVersion() throws SQLException;

    int getDriverMajorVersion();

    int getDriverMinorVersion();

    boolean usesLocalFiles() throws SQLException;

    boolean usesLocalFilePerTable() throws SQLException;

    boolean supportsMixedCaseIdentifiers() throws SQLException;

    boolean storesUpperCaseIdentifiers() throws SQLException;

    boolean storesLowerCaseIdentifiers() throws SQLException;

    boolean storesMixedCaseIdentifiers() throws SQLException;

    boolean supportsMixedCaseQuotedIdentifiers() throws SQLException;

    boolean storesUpperCaseQuotedIdentifiers() throws SQLException;

    boolean storesLowerCaseQuotedIdentifiers() throws SQLException;

    boolean storesMixedCaseQuotedIdentifiers() throws SQLException;

    String getIdentifierQuoteString() throws SQLException;

    String getSQLKeywords() throws SQLException;

    String getNumericFunctions() throws SQLException;

    String getStringFunctions() throws SQLException;

    String getSystemFunctions() throws SQLException;

    String getTimeDateFunctions() throws SQLException;

    String getSearchStringEscape() throws SQLException;

    String getExtraNameCharacters() throws SQLException;

    boolean supportsAlterTableWithAddColumn() throws SQLException;

    boolean supportsAlterTableWithDropColumn() throws SQLException;

    boolean supportsColumnAliasing() throws SQLException;

    boolean nullPlusNonNullIsNull() throws SQLException;

    boolean supportsConvert() throws SQLException;

    boolean supportsConvert(int fromType, int toType) throws SQLException;

    boolean supportsTableCorrelationNames() throws SQLException;

    boolean supportsDifferentTableCorrelationNames() throws SQLException;

    boolean supportsExpressionsInOrderBy() throws SQLException;

    boolean supportsOrderByUnrelated() throws SQLException;

    boolean supportsGroupBy() throws SQLException;

    boolean supportsGroupByUnrelated() throws SQLException;

    boolean supportsGroupByBeyondSelect() throws SQLException;

    boolean supportsLikeEscapeClause() throws SQLException;

    boolean supportsMultipleResultSets() throws SQLException;

    boolean supportsMultipleTransactions() throws SQLException;

    boolean supportsNonNullableColumns() throws SQLException;

    boolean supportsMinimumSQLGrammar() throws SQLException;

    boolean supportsCoreSQLGrammar() throws SQLException;

    boolean supportsExtendedSQLGrammar() throws SQLException;

    boolean supportsANSI92EntryLevelSQL() throws SQLException;

    boolean supportsANSI92IntermediateSQL() throws SQLException;

    boolean supportsANSI92FullSQL() throws SQLException;

    boolean supportsIntegrityEnhancementFacility() throws SQLException;

    boolean supportsOuterJoins() throws SQLException;

    boolean supportsFullOuterJoins() throws SQLException;

    boolean supportsLimitedOuterJoins() throws SQLException;

    String getSchemaTerm() throws SQLException;

    String getProcedureTerm() throws SQLException;

    String getCatalogTerm() throws SQLException;

    boolean isCatalogAtStart() throws SQLException;

    String getCatalogSeparator() throws SQLException;

    boolean supportsSchemasInDataManipulation() throws SQLException;

    boolean supportsSchemasInProcedureCalls() throws SQLException;

    boolean supportsSchemasInTableDefinitions() throws SQLException;

    boolean supportsSchemasInIndexDefinitions() throws SQLException;

    boolean supportsSchemasInPrivilegeDefinitions() throws SQLException;

    boolean supportsCatalogsInDataManipulation() throws SQLException;

    boolean supportsCatalogsInProcedureCalls() throws SQLException;

    boolean supportsCatalogsInTableDefinitions() throws SQLException;

    boolean supportsCatalogsInIndexDefinitions() throws SQLException;

    boolean supportsCatalogsInPrivilegeDefinitions() throws SQLException;

    boolean supportsPositionedDelete() throws SQLException;

    boolean supportsPositionedUpdate() throws SQLException;

    boolean supportsSelectForUpdate() throws SQLException;

    boolean supportsStoredProcedures() throws SQLException;

    boolean supportsSubqueriesInComparisons() throws SQLException;

    boolean supportsSubqueriesInExists() throws SQLException;

    boolean supportsSubqueriesInIns() throws SQLException;

    boolean supportsSubqueriesInQuantifieds() throws SQLException;

    boolean supportsCorrelatedSubqueries() throws SQLException;

    boolean supportsUnion() throws SQLException;

    boolean supportsUnionAll() throws SQLException;

    boolean supportsOpenCursorsAcrossCommit() throws SQLException;

    boolean supportsOpenCursorsAcrossRollback() throws SQLException;

    boolean supportsOpenStatementsAcrossCommit() throws SQLException;

    boolean supportsOpenStatementsAcrossRollback() throws SQLException;

    int getMaxBinaryLiteralLength() throws SQLException;

    int getMaxCharLiteralLength() throws SQLException;

    int getMaxColumnNameLength() throws SQLException;

    int getMaxColumnsInGroupBy() throws SQLException;

    int getMaxColumnsInIndex() throws SQLException;

    int getMaxColumnsInOrderBy() throws SQLException;

    int getMaxColumnsInSelect() throws SQLException;

    int getMaxColumnsInTable() throws SQLException;

    int getMaxConnections() throws SQLException;

    int getMaxCursorNameLength() throws SQLException;

    int getMaxIndexLength() throws SQLException;

    int getMaxSchemaNameLength() throws SQLException;

    int getMaxProcedureNameLength() throws SQLException;

    int getMaxCatalogNameLength() throws SQLException;

    int getMaxRowSize() throws SQLException;

    boolean doesMaxRowSizeIncludeBlobs() throws SQLException;

    int getMaxStatementLength() throws SQLException;

    int getMaxStatements() throws SQLException;

    int getMaxTableNameLength() throws SQLException;

    int getMaxTablesInSelect() throws SQLException;

    int getMaxUserNameLength() throws SQLException;

    int getDefaultTransactionIsolation() throws SQLException;

    boolean supportsTransactions() throws SQLException;

    boolean supportsTransactionIsolationLevel(int level) throws SQLException;

    boolean supportsDataDefinitionAndDataManipulationTransactions() throws SQLException;

    boolean supportsDataManipulationTransactionsOnly() throws SQLException;

    boolean dataDefinitionCausesTransactionCommit() throws SQLException;

    boolean dataDefinitionIgnoredInTransactions() throws SQLException;

    ResultSet getProcedures(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String procedureNamePattern) throws SQLException;

    int procedureResultUnknown;

    int procedureNoResult;

    int procedureReturnsResult;

    ResultSet getProcedureColumns(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String procedureNamePattern, @Nullable String columnNamePattern) throws SQLException;

    int procedureColumnUnknown;

    int procedureColumnIn;

    int procedureColumnInOut;

    int procedureColumnOut;

    int procedureColumnReturn;

    int procedureColumnResult;

    int procedureNoNulls;

    int procedureNullable;

    int procedureNullableUnknown;

    ResultSet getTables(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String tableNamePattern, String @Nullable [] types) throws SQLException;

    ResultSet getSchemas() throws SQLException;

    ResultSet getCatalogs() throws SQLException;

    ResultSet getTableTypes() throws SQLException;

    ResultSet getColumns(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String tableNamePattern, @Nullable String columnNamePattern) throws SQLException;

    int columnNoNulls;

    int columnNullable;

    int columnNullableUnknown;

    ResultSet getColumnPrivileges(@Nullable String catalog, @Nullable String schema, String table, @Nullable String columnNamePattern) throws SQLException;

    ResultSet getTablePrivileges(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String tableNamePattern) throws SQLException;

    ResultSet getBestRowIdentifier(@Nullable String catalog, @Nullable String schema, String table, int scope, boolean nullable) throws SQLException;

    int bestRowTemporary;

    int bestRowTransaction;

    int bestRowSession;

    int bestRowUnknown;

    int bestRowNotPseudo;

    int bestRowPseudo;

    ResultSet getVersionColumns(@Nullable String catalog, @Nullable String schema, String table) throws SQLException;

    int versionColumnUnknown;

    int versionColumnNotPseudo;

    int versionColumnPseudo;

    ResultSet getPrimaryKeys(@Nullable String catalog, @Nullable String schema, String table) throws SQLException;

    ResultSet getImportedKeys(@Nullable String catalog, @Nullable String schema, String table) throws SQLException;

    int importedKeyCascade;

    int importedKeyRestrict;

    int importedKeySetNull;

    int importedKeyNoAction;

    int importedKeySetDefault;

    int importedKeyInitiallyDeferred;

    int importedKeyInitiallyImmediate;

    int importedKeyNotDeferrable;

    ResultSet getExportedKeys(@Nullable String catalog, @Nullable String schema, String table) throws SQLException;

    ResultSet getCrossReference(@Nullable String parentCatalog, @Nullable String parentSchema, String parentTable, @Nullable String foreignCatalog, @Nullable String foreignSchema, String foreignTable) throws SQLException;

    ResultSet getTypeInfo() throws SQLException;

    int typeNoNulls;

    int typeNullable;

    int typeNullableUnknown;

    int typePredNone;

    int typePredChar;

    int typePredBasic;

    int typeSearchable;

    ResultSet getIndexInfo(@Nullable String catalog, @Nullable String schema, String table, boolean unique, boolean approximate) throws SQLException;

    short tableIndexStatistic;

    short tableIndexClustered;

    short tableIndexHashed;

    short tableIndexOther;

    boolean supportsResultSetType(int type) throws SQLException;

    boolean supportsResultSetConcurrency(int type, int concurrency) throws SQLException;

    boolean ownUpdatesAreVisible(int type) throws SQLException;

    boolean ownDeletesAreVisible(int type) throws SQLException;

    boolean ownInsertsAreVisible(int type) throws SQLException;

    boolean othersUpdatesAreVisible(int type) throws SQLException;

    boolean othersDeletesAreVisible(int type) throws SQLException;

    boolean othersInsertsAreVisible(int type) throws SQLException;

    boolean updatesAreDetected(int type) throws SQLException;

    boolean deletesAreDetected(int type) throws SQLException;

    boolean insertsAreDetected(int type) throws SQLException;

    boolean supportsBatchUpdates() throws SQLException;

    ResultSet getUDTs(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String typeNamePattern, int @Nullable [] types) throws SQLException;

    Connection getConnection() throws SQLException;

    boolean supportsSavepoints() throws SQLException;

    boolean supportsNamedParameters() throws SQLException;

    boolean supportsMultipleOpenResults() throws SQLException;

    boolean supportsGetGeneratedKeys() throws SQLException;

    ResultSet getSuperTypes(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String typeNamePattern) throws SQLException;

    ResultSet getSuperTables(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String tableNamePattern) throws SQLException;

    short attributeNoNulls;

    short attributeNullable;

    short attributeNullableUnknown;

    ResultSet getAttributes(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String typeNamePattern, @Nullable String attributeNamePattern) throws SQLException;

    boolean supportsResultSetHoldability(int holdability) throws SQLException;

    int getResultSetHoldability() throws SQLException;

    int getDatabaseMajorVersion() throws SQLException;

    int getDatabaseMinorVersion() throws SQLException;

    int getJDBCMajorVersion() throws SQLException;

    int getJDBCMinorVersion() throws SQLException;

    int sqlStateXOpen;

    int sqlStateSQL;

    int sqlStateSQL99;

    int getSQLStateType() throws SQLException;

    boolean locatorsUpdateCopy() throws SQLException;

    boolean supportsStatementPooling() throws SQLException;

    RowIdLifetime getRowIdLifetime() throws SQLException;

    ResultSet getSchemas(@Nullable String catalog, String schemaPattern) throws SQLException;

    boolean supportsStoredFunctionsUsingCallSyntax() throws SQLException;

    boolean autoCommitFailureClosesAllResultSets() throws SQLException;

    ResultSet getClientInfoProperties() throws SQLException;

    ResultSet getFunctions(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String functionNamePattern) throws SQLException;

    ResultSet getFunctionColumns(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String functionNamePattern, @Nullable String columnNamePattern) throws SQLException;

    int functionColumnUnknown;

    int functionColumnIn;

    int functionColumnInOut;

    int functionColumnOut;

    int functionReturn;

    int functionColumnResult;

    int functionNoNulls;

    int functionNullable;

    int functionNullableUnknown;

    int functionResultUnknown;

    int functionNoTable;

    int functionReturnsTable;

    ResultSet getPseudoColumns(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String tableNamePattern, @Nullable String columnNamePattern) throws SQLException;

    boolean generatedKeyAlwaysReturned() throws SQLException;

    default long getMaxLogicalLobSize() throws SQLException;

    default boolean supportsRefCursors() throws SQLException;

    default boolean supportsSharding() throws SQLException;
}
