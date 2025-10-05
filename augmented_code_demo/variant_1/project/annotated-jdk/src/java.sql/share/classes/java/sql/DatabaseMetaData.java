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
package java.sql;

    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;

    @Positive
@AnnotatedFor("nullness")
    @Positive
public interface DatabaseMetaData extends Wrapper {

    @Positive
    boolean allProceduresAreCallable() throws SQLException;

    @Positive
    boolean allTablesAreSelectable() throws SQLException;

    @Positive
    @Nullable
    @Positive
    String getURL() throws SQLException;

    @Positive
    String getUserName() throws SQLException;

    @Positive
    boolean isReadOnly() throws SQLException;

    @Positive
    boolean nullsAreSortedHigh() throws SQLException;

    @Positive
    boolean nullsAreSortedLow() throws SQLException;

    @Positive
    boolean nullsAreSortedAtStart() throws SQLException;

    @Positive
    boolean nullsAreSortedAtEnd() throws SQLException;

    @Positive
    String getDatabaseProductName() throws SQLException;

    @Positive
    String getDatabaseProductVersion() throws SQLException;

    @Positive
    String getDriverName() throws SQLException;

    @Positive
    String getDriverVersion() throws SQLException;

    @Positive
    int getDriverMajorVersion();

    @Positive
    int getDriverMinorVersion();

    @Positive
    boolean usesLocalFiles() throws SQLException;

    @Positive
    boolean usesLocalFilePerTable() throws SQLException;

    @Positive
    boolean supportsMixedCaseIdentifiers() throws SQLException;

    @Positive
    boolean storesUpperCaseIdentifiers() throws SQLException;

    @Positive
    boolean storesLowerCaseIdentifiers() throws SQLException;

    @Positive
    boolean storesMixedCaseIdentifiers() throws SQLException;

    @Positive
    boolean supportsMixedCaseQuotedIdentifiers() throws SQLException;

    @Positive
    boolean storesUpperCaseQuotedIdentifiers() throws SQLException;

    @Positive
    boolean storesLowerCaseQuotedIdentifiers() throws SQLException;

    @Positive
    boolean storesMixedCaseQuotedIdentifiers() throws SQLException;

    @Positive
    String getIdentifierQuoteString() throws SQLException;

    @Positive
    String getSQLKeywords() throws SQLException;

    @Positive
    String getNumericFunctions() throws SQLException;

    @Positive
    String getStringFunctions() throws SQLException;

    @Positive
    String getSystemFunctions() throws SQLException;

    @Positive
    String getTimeDateFunctions() throws SQLException;

    @Positive
    String getSearchStringEscape() throws SQLException;

    @Positive
    String getExtraNameCharacters() throws SQLException;

    @Positive
    boolean supportsAlterTableWithAddColumn() throws SQLException;

    @Positive
    boolean supportsAlterTableWithDropColumn() throws SQLException;

    @Positive
    boolean supportsColumnAliasing() throws SQLException;

    @Positive
    boolean nullPlusNonNullIsNull() throws SQLException;

    @Positive
    boolean supportsConvert() throws SQLException;

    @Positive
    boolean supportsConvert(int fromType, int toType) throws SQLException;

    @Positive
    boolean supportsTableCorrelationNames() throws SQLException;

    @Positive
    boolean supportsDifferentTableCorrelationNames() throws SQLException;

    @Positive
    boolean supportsExpressionsInOrderBy() throws SQLException;

    @Positive
    boolean supportsOrderByUnrelated() throws SQLException;

    @Positive
    boolean supportsGroupBy() throws SQLException;

    @Positive
    boolean supportsGroupByUnrelated() throws SQLException;

    @Positive
    boolean supportsGroupByBeyondSelect() throws SQLException;

    @Positive
    boolean supportsLikeEscapeClause() throws SQLException;

    @Positive
    boolean supportsMultipleResultSets() throws SQLException;

    @Positive
    boolean supportsMultipleTransactions() throws SQLException;

    @Positive
    boolean supportsNonNullableColumns() throws SQLException;

    @Positive
    boolean supportsMinimumSQLGrammar() throws SQLException;

    @Positive
    boolean supportsCoreSQLGrammar() throws SQLException;

    @Positive
    boolean supportsExtendedSQLGrammar() throws SQLException;

    @Positive
    boolean supportsANSI92EntryLevelSQL() throws SQLException;

    @Positive
    boolean supportsANSI92IntermediateSQL() throws SQLException;

    @Positive
    boolean supportsANSI92FullSQL() throws SQLException;

    @Positive
    boolean supportsIntegrityEnhancementFacility() throws SQLException;

    @Positive
    boolean supportsOuterJoins() throws SQLException;

    @Positive
    boolean supportsFullOuterJoins() throws SQLException;

    @Positive
    boolean supportsLimitedOuterJoins() throws SQLException;

    @Positive
    String getSchemaTerm() throws SQLException;

    @Positive
    String getProcedureTerm() throws SQLException;

    @Positive
    String getCatalogTerm() throws SQLException;

    @Positive
    boolean isCatalogAtStart() throws SQLException;

    @Positive
    String getCatalogSeparator() throws SQLException;

    @Positive
    boolean supportsSchemasInDataManipulation() throws SQLException;

    @Positive
    boolean supportsSchemasInProcedureCalls() throws SQLException;

    @Positive
    boolean supportsSchemasInTableDefinitions() throws SQLException;

    @Positive
    boolean supportsSchemasInIndexDefinitions() throws SQLException;

    @Positive
    boolean supportsSchemasInPrivilegeDefinitions() throws SQLException;

    @Positive
    boolean supportsCatalogsInDataManipulation() throws SQLException;

    @Positive
    boolean supportsCatalogsInProcedureCalls() throws SQLException;

    @Positive
    boolean supportsCatalogsInTableDefinitions() throws SQLException;

    @Positive
    boolean supportsCatalogsInIndexDefinitions() throws SQLException;

    @Positive
    boolean supportsCatalogsInPrivilegeDefinitions() throws SQLException;

    @Positive
    boolean supportsPositionedDelete() throws SQLException;

    @Positive
    boolean supportsPositionedUpdate() throws SQLException;

    @Positive
    boolean supportsSelectForUpdate() throws SQLException;

    @Positive
    boolean supportsStoredProcedures() throws SQLException;

    @Positive
    boolean supportsSubqueriesInComparisons() throws SQLException;

    @Positive
    boolean supportsSubqueriesInExists() throws SQLException;

    @Positive
    boolean supportsSubqueriesInIns() throws SQLException;

    @Positive
    boolean supportsSubqueriesInQuantifieds() throws SQLException;

    @Positive
    boolean supportsCorrelatedSubqueries() throws SQLException;

    @Positive
    boolean supportsUnion() throws SQLException;

    @Positive
    boolean supportsUnionAll() throws SQLException;

    @Positive
    boolean supportsOpenCursorsAcrossCommit() throws SQLException;

    @Positive
    boolean supportsOpenCursorsAcrossRollback() throws SQLException;

    @Positive
    boolean supportsOpenStatementsAcrossCommit() throws SQLException;

    @Positive
    boolean supportsOpenStatementsAcrossRollback() throws SQLException;

    @Positive
    int getMaxBinaryLiteralLength() throws SQLException;

    @Positive
    int getMaxCharLiteralLength() throws SQLException;

    @Positive
    int getMaxColumnNameLength() throws SQLException;

    @Positive
    int getMaxColumnsInGroupBy() throws SQLException;

    @Positive
    int getMaxColumnsInIndex() throws SQLException;

    @Positive
    int getMaxColumnsInOrderBy() throws SQLException;

    @Positive
    int getMaxColumnsInSelect() throws SQLException;

    @Positive
    int getMaxColumnsInTable() throws SQLException;

    @Positive
    int getMaxConnections() throws SQLException;

    @Positive
    int getMaxCursorNameLength() throws SQLException;

    @Positive
    int getMaxIndexLength() throws SQLException;

    @Positive
    int getMaxSchemaNameLength() throws SQLException;

    @Positive
    int getMaxProcedureNameLength() throws SQLException;

    @Positive
    int getMaxCatalogNameLength() throws SQLException;

    @Positive
    int getMaxRowSize() throws SQLException;

    @Positive
    boolean doesMaxRowSizeIncludeBlobs() throws SQLException;

    @Positive
    int getMaxStatementLength() throws SQLException;

    @Positive
    int getMaxStatements() throws SQLException;

    @Positive
    int getMaxTableNameLength() throws SQLException;

    @Positive
    int getMaxTablesInSelect() throws SQLException;

    @Positive
    int getMaxUserNameLength() throws SQLException;

    @Positive
    int getDefaultTransactionIsolation() throws SQLException;

    @Positive
    boolean supportsTransactions() throws SQLException;

    @Positive
    boolean supportsTransactionIsolationLevel(int level) throws SQLException;

    @Positive
    boolean supportsDataDefinitionAndDataManipulationTransactions() throws SQLException;

    @Positive
    boolean supportsDataManipulationTransactionsOnly() throws SQLException;

    @Positive
    boolean dataDefinitionCausesTransactionCommit() throws SQLException;

    @Positive
    boolean dataDefinitionIgnoredInTransactions() throws SQLException;

    @Positive
    ResultSet getProcedures(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String procedureNamePattern) throws SQLException;

    @Positive
    int procedureResultUnknown;

    @Positive
    int procedureNoResult;

    @Positive
    int procedureReturnsResult;

    @Positive
    ResultSet getProcedureColumns(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String procedureNamePattern, @Nullable String columnNamePattern) throws SQLException;

    @Positive
    int procedureColumnUnknown;

    @Positive
    int procedureColumnIn;

    @Positive
    int procedureColumnInOut;

    @Positive
    int procedureColumnOut;

    @Positive
    int procedureColumnReturn;

    @Positive
    int procedureColumnResult;

    @Positive
    int procedureNoNulls;

    @Positive
    int procedureNullable;

    @Positive
    int procedureNullableUnknown;

    @Positive
    ResultSet getTables(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String tableNamePattern, String @Nullable [] types) throws SQLException;

    @Positive
    ResultSet getSchemas() throws SQLException;

    @Positive
    ResultSet getCatalogs() throws SQLException;

    @Positive
    ResultSet getTableTypes() throws SQLException;

    @Positive
    ResultSet getColumns(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String tableNamePattern, @Nullable String columnNamePattern) throws SQLException;

    @Positive
    int columnNoNulls;

    @Positive
    int columnNullable;

    @Positive
    int columnNullableUnknown;

    @Positive
    ResultSet getColumnPrivileges(@Nullable String catalog, @Nullable String schema, String table, @Nullable String columnNamePattern) throws SQLException;

    @Positive
    ResultSet getTablePrivileges(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String tableNamePattern) throws SQLException;

    @Positive
    ResultSet getBestRowIdentifier(@Nullable String catalog, @Nullable String schema, String table, int scope, boolean nullable) throws SQLException;

    @Positive
    int bestRowTemporary;

    @Positive
    int bestRowTransaction;

    @Positive
    int bestRowSession;

    @Positive
    int bestRowUnknown;

    @Positive
    int bestRowNotPseudo;

    @Positive
    int bestRowPseudo;

    @Positive
    ResultSet getVersionColumns(@Nullable String catalog, @Nullable String schema, String table) throws SQLException;

    @Positive
    int versionColumnUnknown;

    @Positive
    int versionColumnNotPseudo;

    @Positive
    int versionColumnPseudo;

    @Positive
    ResultSet getPrimaryKeys(@Nullable String catalog, @Nullable String schema, String table) throws SQLException;

    @Positive
    ResultSet getImportedKeys(@Nullable String catalog, @Nullable String schema, String table) throws SQLException;

    @Positive
    int importedKeyCascade;

    @Positive
    int importedKeyRestrict;

    @Positive
    int importedKeySetNull;

    @Positive
    int importedKeyNoAction;

    @Positive
    int importedKeySetDefault;

    @Positive
    int importedKeyInitiallyDeferred;

    @Positive
    int importedKeyInitiallyImmediate;

    @Positive
    int importedKeyNotDeferrable;

    @Positive
    ResultSet getExportedKeys(@Nullable String catalog, @Nullable String schema, String table) throws SQLException;

    @Positive
    ResultSet getCrossReference(@Nullable String parentCatalog, @Nullable String parentSchema, String parentTable, @Nullable String foreignCatalog, @Nullable String foreignSchema, String foreignTable) throws SQLException;

    @Positive
    ResultSet getTypeInfo() throws SQLException;

    @Positive
    int typeNoNulls;

    @Positive
    int typeNullable;

    @Positive
    int typeNullableUnknown;

    @Positive
    int typePredNone;

    @Positive
    int typePredChar;

    @Positive
    int typePredBasic;

    @Positive
    int typeSearchable;

    @Positive
    ResultSet getIndexInfo(@Nullable String catalog, @Nullable String schema, String table, boolean unique, boolean approximate) throws SQLException;

    @Positive
    short tableIndexStatistic;

    @Positive
    short tableIndexClustered;

    @Positive
    short tableIndexHashed;

    @Positive
    short tableIndexOther;

    @Positive
    boolean supportsResultSetType(int type) throws SQLException;

    @Positive
    boolean supportsResultSetConcurrency(int type, int concurrency) throws SQLException;

    @Positive
    boolean ownUpdatesAreVisible(int type) throws SQLException;

    @Positive
    boolean ownDeletesAreVisible(int type) throws SQLException;

    @Positive
    boolean ownInsertsAreVisible(int type) throws SQLException;

    @Positive
    boolean othersUpdatesAreVisible(int type) throws SQLException;

    @Positive
    boolean othersDeletesAreVisible(int type) throws SQLException;

    @Positive
    boolean othersInsertsAreVisible(int type) throws SQLException;

    @Positive
    boolean updatesAreDetected(int type) throws SQLException;

    @Positive
    boolean deletesAreDetected(int type) throws SQLException;

    @Positive
    boolean insertsAreDetected(int type) throws SQLException;

    @Positive
    boolean supportsBatchUpdates() throws SQLException;

    @Positive
    ResultSet getUDTs(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String typeNamePattern, int @Nullable [] types) throws SQLException;

    @Positive
    Connection getConnection() throws SQLException;

    @Positive
    boolean supportsSavepoints() throws SQLException;

    @Positive
    boolean supportsNamedParameters() throws SQLException;

    @Positive
    boolean supportsMultipleOpenResults() throws SQLException;

    @Positive
    boolean supportsGetGeneratedKeys() throws SQLException;

    @Positive
    ResultSet getSuperTypes(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String typeNamePattern) throws SQLException;

    @Positive
    ResultSet getSuperTables(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String tableNamePattern) throws SQLException;

    @Positive
    short attributeNoNulls;

    @Positive
    short attributeNullable;

    @Positive
    short attributeNullableUnknown;

    @Positive
    ResultSet getAttributes(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String typeNamePattern, @Nullable String attributeNamePattern) throws SQLException;

    @Positive
    boolean supportsResultSetHoldability(int holdability) throws SQLException;

    @Positive
    int getResultSetHoldability() throws SQLException;

    @Positive
    int getDatabaseMajorVersion() throws SQLException;

    @Positive
    int getDatabaseMinorVersion() throws SQLException;

    @Positive
    int getJDBCMajorVersion() throws SQLException;

    @Positive
    int getJDBCMinorVersion() throws SQLException;

    @Positive
    int sqlStateXOpen;

    @Positive
    int sqlStateSQL;

    @Positive
    int sqlStateSQL99;

    @Positive
    int getSQLStateType() throws SQLException;

    @Positive
    boolean locatorsUpdateCopy() throws SQLException;

    @Positive
    boolean supportsStatementPooling() throws SQLException;

    @Positive
    RowIdLifetime getRowIdLifetime() throws SQLException;

    @Positive
    ResultSet getSchemas(@Nullable String catalog, String schemaPattern) throws SQLException;

    @Positive
    boolean supportsStoredFunctionsUsingCallSyntax() throws SQLException;

    @Positive
    boolean autoCommitFailureClosesAllResultSets() throws SQLException;

    @Positive
    ResultSet getClientInfoProperties() throws SQLException;

    @Positive
    ResultSet getFunctions(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String functionNamePattern) throws SQLException;

    @Positive
    ResultSet getFunctionColumns(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String functionNamePattern, @Nullable String columnNamePattern) throws SQLException;

    @Positive
    int functionColumnUnknown;

    @Positive
    int functionColumnIn;

    @Positive
    int functionColumnInOut;

    @Positive
    int functionColumnOut;

    @Positive
    int functionReturn;

    @Positive
    int functionColumnResult;

    @Positive
    int functionNoNulls;

    @Positive
    int functionNullable;

    @Positive
    int functionNullableUnknown;

    @Positive
    int functionResultUnknown;

    @Positive
    int functionNoTable;

    @Positive
    int functionReturnsTable;

    @Positive
    ResultSet getPseudoColumns(@Nullable String catalog, @Nullable String schemaPattern, @Nullable String tableNamePattern, @Nullable String columnNamePattern) throws SQLException;

    @Positive
    boolean generatedKeyAlwaysReturned() throws SQLException;

    @Positive
    default long getMaxLogicalLobSize() throws SQLException;

    @Positive
    default boolean supportsRefCursors() throws SQLException;

    @Positive
    default boolean supportsSharding() throws SQLException;
    @Positive
}

// CFWR semantic augmentation - variant 1
